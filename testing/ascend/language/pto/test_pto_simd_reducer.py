"""PTO reducer state, horizontal finalization and one-lane consumers."""
import pytest
import tilelang.ascend.language as T
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower

pytestmark = pytest.mark.pto


def reducer_kernel(n=257, lanes=128, dtype="float32", op="sum", seed=False, arithmetic=False, conditional=False):
    padded = (n + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((padded,), dtype), Out: T.Tensor((padded,), dtype), Result: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), dtype)
            b = T.alloc_shared((padded,), dtype)
            out = T.alloc_shared((lanes,), dtype)
            T.copy(A, a)
            T.copy(Out, b)
            T.copy(Result, out)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_reducer((1,), dtype, op=op)
                result = T.alloc_local((1,), dtype)
                if seed:
                    T.reducer_init(acc, T.Cast(dtype, 7))
                else:
                    T.reducer_init(acc)
                for i in T.Parallel(n):
                    if conditional:
                        if a[i] > T.Cast(dtype, 0):
                            T.reducer_update(acc[0], a[i])
                    else:
                        T.reducer_update(acc[0], a[i])
                T.finalize_reducer(acc, result)
                alias = result[0]
                out[0] = alias
                for i in T.Parallel(n):
                    if arithmetic:
                        b[i] = a[i] * (alias / T.Cast(dtype, 4096))
                    else:
                        b[i] = a[i] * alias
            T.copy(b, Out)
            T.copy(out, Result)
    return main


def _lower(f):
    target = normalize_pto_target("pto")
    with target:
        return lower(f, target=target)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_reducer_codegen(dtype, op):
    source = _lower(reducer_kernel(dtype=dtype, op=op)).kernel_source
    assert {"sum": "pto.vmi.vcadd", "max": "pto.vmi.vcmax", "min": "pto.vmi.vcmin"}[op] in source
    assert "group=1" in source
    assert "pto.vmi.vsel" in source


def test_seed_and_one_lane_arithmetic():
    source = _lower(reducer_kernel(seed=True, arithmetic=True)).kernel_source
    assert "pto.vmi.vdiv" in source


@pytest.mark.parametrize("lanes", [64, 128, 256])
@pytest.mark.parametrize("extent", ["one", "short", "full", "tail"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_reducer_layout_width(lanes, extent, dtype, op):
    n = {"one": 1, "short": 17, "full": lanes, "tail": 2 * lanes + 1}[extent]
    assert {"sum": "vcadd", "max": "vcmax", "min": "vcmin"}[op] in _lower(reducer_kernel(n=n, lanes=lanes, dtype=dtype, op=op)).kernel_source


def unsupported_kernel(updates=1, arithmetic="none", seed=False):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), Out: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                acc = T.alloc_reducer((1,), "float32", op="sum")
                result = T.alloc_local((1,), "float32")
                if arithmetic == "seed":
                    T.reducer_init(acc, a[0] + 1)
                elif seed:
                    T.reducer_init(acc, T.float32(7))
                else:
                    T.reducer_init(acc)
                if updates > 0:
                    for i in T.Parallel(128):
                        T.reducer_update(acc[0], a[i])
                        if updates > 1:
                            T.reducer_update(acc[0], a[i])
                T.finalize_reducer(acc, result)
                if arithmetic == "store":
                    a[0] = result[0] / 4096
                elif arithmetic == "bind":
                    bad = result[0] / 4096
                    a[0] = bad
                else:
                    a[0] = result[0]
            T.copy(a, Out)
    return main


@pytest.mark.parametrize("updates,seed", [(0, False), (0, True), (2, False)])
def test_update_count_diagnostic(updates, seed):
    with pytest.raises(Exception, match="exactly one static update"):
        _lower(unsupported_kernel(updates=updates, seed=seed))


@pytest.mark.parametrize("arithmetic", ["seed", "store", "bind"])
def test_region_arithmetic_rejected(arithmetic):
    with pytest.raises(Exception, match="ordinary data arithmetic outside Parallel"):
        _lower(unsupported_kernel(arithmetic=arithmetic))


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_conditional_reducer(op):
    assert "pto.vmi.vcmp" in _lower(reducer_kernel(op=op, conditional=True)).kernel_source


def seed_snapshot_kernel(vmi_seed=False, repeat=1):
    @T.prim_func
    def main(A: T.Tensor((384,), "float32"), Result: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((384,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                source = T.alloc_local((1,), "float32")
                result = T.alloc_local((1,), "float32")
                if vmi_seed:
                    first = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(first)
                    for j in T.Parallel(1):
                        T.reducer_update(first[0], a[j])
                    T.finalize_reducer(first, source)
                else:
                    source[0] = T.float32(7)
                for epoch in T.serial(repeat):
                    acc = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(acc, source[0])
                    source[0] = T.float32(19)
                    for i in T.Parallel(257):
                        T.reducer_update(acc[0], a[i])
                    T.finalize_reducer(acc, result)
                    out[epoch] = result[0]
            T.copy(out, Result)
    return main


@pytest.mark.parametrize("vmi_seed", [False, True])
def test_seed_snapshot_codegen(vmi_seed):
    source = _lower(seed_snapshot_kernel(vmi_seed)).kernel_source
    assert "pto.vmi.vcadd" in source


def test_repeated_epoch_codegen():
    assert "pto.vmi.vcadd" in _lower(seed_snapshot_kernel(repeat=2)).kernel_source


def mixed_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), Out: T.Tensor((256,), "float32"), Result: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((256,), "float32")
            reduced = T.alloc_shared((1,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    b[i] = a[i] * 2
                T.reduce_sum(b, reduced, dim=0)
                acc = T.alloc_reducer((1,), "float32", op="sum")
                result = T.alloc_local((1,), "float32")
                T.reducer_init(acc)
                for i in T.Parallel(256):
                    T.reducer_update(acc[0], b[i])
                T.finalize_reducer(acc, result)
                out[0] = result[0]
                for i in T.Parallel(256):
                    b[i] = a[i] * result[0] + reduced[0]
            T.copy(b, Out)
            T.copy(out, Result)
    return main


def test_mixed_direct_reduce():
    assert _lower(mixed_kernel()).kernel_source.count("pto.vmi.vcadd") == 2


def _capture_stages(f, monkeypatch):
    from tilelang import transform
    from tilelang.ascend import transform as ascend
    snapshots = {}
    for owner, name in [(ascend, "AscendLayoutInference"), (transform, "ReducerPlanAndMaterialize"),
                        (ascend, "VectorizeParallelToPTO")]:
        if not hasattr(owner, name):
            continue
        original = getattr(owner, name)
        def factory(*args, _original=original, _name=name, **kwargs):
            run = _original(*args, **kwargs)
            def wrapped(mod):
                snapshots[_name + "_before"] = mod
                result = run(mod)
                snapshots[_name] = result
                return result
            return wrapped
        monkeypatch.setattr(owner, name, factory)
    _lower(f)
    return snapshots


def test_finalize_dataflow(monkeypatch):
    from tilelang import tvm
    from testing.ascend._ir import calls, nodes
    s = _capture_stages(reducer_kernel(seed=True, arithmetic=True), monkeypatch)
    layout_vf = next(b for b in nodes(s["AscendLayoutInference"], tvm.tirx.SBlock) if b.name_hint == "SIMD_VF")
    layouts = layout_vf.annotations["layout_map"]
    partials = [(b, layout) for b, layout in layouts.items() if "reducer" in b.scope()]
    assert len(partials) == 1 and "PartialFragment" in str(type(partials[0][1]))
    assert not any(b.name == "result" for b in layouts)
    before = s["VectorizeParallelToPTO_before"]
    after = s["VectorizeParallelToPTO"]
    vf = next(b for b in nodes(before, tvm.tirx.SBlock) if b.name_hint == "SIMD_VF")
    states = vf.annotations["tl.pto.reducer_states"]
    assert len(states) == 1
    assert any(b.data.same_as(states[0]) and b.scope() == "local" for b in vf.alloc_buffers)
    final = calls(before, "tl.tileop.pto_finalize_reducer")
    assert len(final) == 1 and [int(x) for x in final[0].annotations["plan_steps"]] == [128, 1]
    assert not calls(after, "tl.tileop.pto_finalize_reducer")
    assert all("tl.pto.reducer_states" not in b.annotations for b in nodes(after, tvm.tirx.SBlock))
    binds = nodes(after, tvm.tirx.Bind)
    producer = {b.var: b.value for b in binds}
    horizontal = calls(after, "tl.vmi.vcadd")[0]
    mask = producer[horizontal.args[1]]
    assert mask.op.name == "tl.vmi.create_mask" and int(mask.args[0]) == 128
    div = calls(after, "tl.vmi.vdiv")[0]
    one_mask = producer[div.args[2]]
    assert int(one_mask.annotations["size"]) == 1 and int(one_mask.args[0]) == 1
    div_var = next(b.var for b in binds if b.value.same_as(div))
    broadcast = next(c for c in calls(after, "tl.vmi.vbrc") if c.args[0].same_as(div_var))
    assert int(broadcast.annotations["group"]) == 1
    assert int(broadcast.annotations["size"]) == 128
    one_stores = [c for c in calls(after, "tl.vmi.vstore") if len(c.args) == 3]
    assert len(one_stores) == 1 and int(one_stores[0].args[1].args[1]) == 1


@pytest.mark.parametrize("damage,match", [
    ("missing", "missing from tl.pto.reducer_states"),
    ("foreign", "allocation"),
    ("odd_plan", "width/scale pairs"),
    ("unsupported_plan", "only plan_steps"),
    ("empty_plan", "only plan_steps"),
])
def test_materialized_contract_diagnostics(monkeypatch, damage, match):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from tvm.tirx.stmt_functor import ir_transform
    s = _capture_stages(reducer_kernel(), monkeypatch)
    mod = s["VectorizeParallelToPTO_before"]
    def rewrite(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            annotations = dict(node.annotations)
            if damage == "missing": annotations.pop("tl.pto.reducer_states")
            if damage == "foreign": annotations["tl.pto.reducer_states"] = [tvm.tirx.Var("foreign", "handle")]
            return tvm.tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint,
                                  node.body, node.init, node.alloc_buffers, node.match_buffers, annotations)
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer" and damage.endswith("plan"):
            annotations = dict(node.annotations)
            annotations["plan_steps"] = [tvm.tirx.IntImm("int32", x) for x in {"odd_plan": [128], "unsupported_plan": [64, 1], "empty_plan": []}[damage]]
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)
    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match=match):
        ascend.VerifyParallelToPTO()(changed)


def test_plan_element_type_checked(monkeypatch):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from tvm.tirx.stmt_functor import ir_transform
    mod = _capture_stages(reducer_kernel(), monkeypatch)["VectorizeParallelToPTO_before"]
    def rewrite(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer":
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations={"plan_steps": ["invalid", "invalid"]})
    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match="plan_steps must be IntImm"):
        ascend.VerifyParallelToPTO()(changed)


def test_handwritten_vmi_without_lanes_unchanged():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF():
                value = T.vmi.vload(a[0], size=128)
                T.vmi.vstore(value, b[0])
            T.copy(b, B)
    assert "pto.vmi.vload" in _lower(main).kernel_source


@pytest.mark.parametrize("damage,match", [
    ("offset", "single local slot"),
    ("overlap", "independent"),
    ("dtype", "dtype mismatch"),
    ("access", "access_mask"),
])
def test_finalize_regions_checked(monkeypatch, damage, match):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from testing.ascend._ir import region
    from tvm.tirx.stmt_functor import ir_transform
    mod = _capture_stages(reducer_kernel(), monkeypatch)["VectorizeParallelToPTO_before"]
    def rewrite(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer":
            args = list(node.args)
            acc = args[0].args[0].buffer
            dst = args[1].args[0].buffer
            if damage == "offset": args[1] = region(dst, 2, [1], [1])
            elif damage == "overlap": args[1] = region(acc, 2)
            elif damage == "dtype":
                bad = tvm.tirx.decl_buffer((1,), "float16", "bad", scope="local")
                args[1] = region(bad, 2)
            else: args[1] = region(dst, 1)
            return tvm.tirx.Call(node.dtype, node.op, args, annotations=node.annotations)
    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match=match):
        ascend.VerifyParallelToPTO()(changed)
