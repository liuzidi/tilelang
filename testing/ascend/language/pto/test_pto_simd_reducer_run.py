"""Required device checks; missing NPU/PTOAS is a failure, never a skip."""
import pytest
import torch
import tilelang
from testing.ascend.language.pto.test_pto_simd_reducer import reducer_kernel, seed_snapshot_kernel

pytestmark = pytest.mark.pto

CASES = [(128, 1), (128, 17), (128, 257), (128, 4096), (64, 129), (256, 513)]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("lanes,n", CASES)
def test_reducer_numerics(dtype, op, lanes, n):
    assert hasattr(torch, "npu"), "NPU runtime is required"
    td = getattr(torch, dtype)
    p = (n + lanes - 1) // lanes * lanes
    cpu = torch.ones(p, dtype=td)
    if op == "max":
        cpu[:n] = -(torch.arange(n) % 13 + 1).to(td)
        cpu[max(0, n - 1)] = -0.5  # Unique extremum in the tail.
        cpu[n:] = 100
        expected = cpu[:n].max()
    elif op == "min":
        cpu[:n] = (torch.arange(n) % 13 + 1).to(td)
        cpu[0] = 0.5  # Unique extremum in the first chunk.
        cpu[n:] = -100
        expected = cpu[:n].min()
    else:
        # Different lane contributions; at most 4096 and exactly representable.
        cpu[:n] = (torch.arange(n) % 2).to(td) if n == 4096 else 1
        cpu[n:] = 37
        expected = cpu[:n].sum(dtype=torch.float64).to(td)
    kernel = tilelang.compile(reducer_kernel(n=n, lanes=lanes, dtype=dtype, op=op), target="pto")
    out = torch.full((p,), -123, dtype=td).npu()
    result = torch.full((lanes,), -123, dtype=td).npu()
    kernel(cpu.npu(), out, result)
    torch.testing.assert_close(result.cpu()[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(result.cpu()[1:], torch.full((lanes - 1,), -123, dtype=td), rtol=0, atol=0)
    expected_out = torch.full((p,), -123, dtype=td)
    expected_out[:n] = cpu[:n] * expected
    torch.testing.assert_close(out.cpu(), expected_out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("n", [257, 4096])
def test_one_lane_division(dtype, n):
    td = getattr(torch, dtype)
    p = (n + 127) // 128 * 128
    a = torch.ones(p, dtype=td).npu()
    out = torch.full((p,), -123, dtype=td).npu()
    result = torch.full((128,), -123, dtype=td).npu()
    kernel = tilelang.compile(reducer_kernel(n=n, dtype=dtype, arithmetic=True), target="pto")
    kernel(a, out, result)
    torch.testing.assert_close(out.cpu()[:n], torch.full((n,), n / 4096, dtype=td), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu()[n:], torch.full((p - n,), -123, dtype=td), rtol=0, atol=0)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("all_false", [False, True])
def test_conditional_and_seed(op, all_false):
    a = torch.full((384,), -2., dtype=torch.float32)
    if not all_false:
        a[::3] = 3
        a[256] = 11
    a[257:] = 91
    selected = a[:257][a[:257] > 0]
    if op == "sum": expected = selected.sum() + 7
    elif op == "max": expected = torch.cat((selected, torch.tensor([7.]))).max()
    else: expected = torch.cat((selected, torch.tensor([7.]))).min()
    kernel = tilelang.compile(reducer_kernel(op=op, seed=True, conditional=True), target="pto")
    out = torch.full((384,), -123.).npu()
    result = torch.full((128,), -123.).npu()
    kernel(a.npu(), out, result)
    torch.testing.assert_close(result.cpu()[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu()[:257], a[:257] * expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu()[257:], torch.full((127,), -123.), rtol=0, atol=0)


@pytest.mark.parametrize("vmi_seed,repeat", [(False, 1), (True, 1), (False, 2)])
def test_seed_capture(vmi_seed, repeat):
    kernel = tilelang.compile(seed_snapshot_kernel(vmi_seed, repeat), target="pto", out_idx=-1)
    actual = kernel(torch.ones(384).npu()).cpu()
    expected = torch.tensor([258.] if vmi_seed else ([264.] if repeat == 1 else [264., 276.]))
    torch.testing.assert_close(actual[:repeat], expected, rtol=0, atol=0)


def test_mixed_direct_reduce_numerics():
    from testing.ascend.language.pto.test_pto_simd_reducer import mixed_kernel
    kernel = tilelang.compile(mixed_kernel(), target="pto")
    a = torch.arange(256, dtype=torch.float32) % 7
    out = torch.empty(256).npu()
    result = torch.empty(128).npu()
    kernel(a.npu(), out, result)
    total = (a * 2).sum()
    torch.testing.assert_close(result.cpu()[0], total, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), a * total + total, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_floating_inputs(dtype, op):
    td = getattr(torch, dtype)
    gen = torch.Generator().manual_seed(20261111)
    a = (torch.rand(384, generator=gen) * 0.5 + 0.25).to(td)
    valid = a[:257].to(torch.float64)
    expected = {"sum": valid.sum, "max": valid.max, "min": valid.min}[op]().to(td)
    a[257:] = 99 if op != "min" else -99
    kernel = tilelang.compile(reducer_kernel(dtype=dtype, op=op), target="pto")
    out = torch.full((384,), -123., dtype=td).npu()
    result = torch.full((128,), -123., dtype=td).npu()
    kernel(a.npu(), out, result)
    tolerance = 0.002 if dtype == "float16" else 0.000002
    torch.testing.assert_close(result.cpu()[0], expected, rtol=tolerance, atol=0)
    torch.testing.assert_close(out.cpu()[:257], a[:257] * expected, rtol=tolerance * 2, atol=0)
    torch.testing.assert_close(out.cpu()[257:], torch.full((127,), -123., dtype=td), rtol=0, atol=0)
