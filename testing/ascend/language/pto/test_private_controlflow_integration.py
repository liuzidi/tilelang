"""Integration probes for PR38 with PR11; mandatory device tests."""
import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from testing.ascend.language.pto.test_simdvf_parallel_control_flow import _lower_capture, _vmi_call_names

pytestmark = pytest.mark.pto


def kernel(kind, mode, n=131):
    p = (n + 63) // 64 * 64

    @T.prim_func
    def main(A: T.Tensor((p,), 'float32'), Seed: T.Tensor((p,), 'float32'), Out: T.Tensor((p,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((p,), 'float32')
            b = T.alloc_shared((p,), 'float32')
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(n):
                    if kind == 'var':
                        acc = T.alloc_var('float32')
                        acc = a[i]
                        old = acc
                        if mode == 'two':
                            if acc > 0:
                                acc = acc + 7.0
                            else:
                                acc = acc - 3.0
                        else:
                            if acc > 0:
                                acc = acc + 7.0
                        b[i] = old * 2.0 + acc
                    else:
                        tmp = T.alloc_local((1,), 'float32')
                        tmp[0] = a[i]
                        old = tmp[0]
                        if mode == 'two':
                            if tmp[0] > 0:
                                tmp[0] = tmp[0] + 7.0
                            else:
                                tmp[0] = tmp[0] - 3.0
                        else:
                            if tmp[0] > 0:
                                tmp[0] = tmp[0] + 7.0
                        b[i] = old * 2.0 + tmp[0]
            T.copy(b, Out)
    return main


@pytest.mark.parametrize('kind', ['var', 'local'])
@pytest.mark.parametrize('mode', ['one', 'two'])
def test_lower(kind, mode):
    stages, error = _lower_capture(kernel(kind, mode), expect_error=True)
    assert error is None, str(error)
    assert 'tl.vmi.vsel' in _vmi_call_names(stages['vectorize'])


@pytest.mark.parametrize('kind', ['var', 'local'])
@pytest.mark.parametrize('mode', ['one', 'two'])
@pytest.mark.parametrize('n', [64, 131])
def test_device(kind, mode, n):
    p = (n + 63) // 64 * 64
    f = tilelang.compile(kernel(kind, mode, n), target='pto', out_idx=-1)
    x = (torch.arange(p, dtype=torch.float32) % 37) - 18
    seed = torch.full((p,), -12345.0)
    expected = seed.clone()
    expected[:n] = 3*x[:n] + torch.where(x[:n] > 0, 7.0, -3.0 if mode == 'two' else 0.0)
    torch.testing.assert_close(f(x.npu(), seed.npu()).cpu(), expected, rtol=0, atol=0)
