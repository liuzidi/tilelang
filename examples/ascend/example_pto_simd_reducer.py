"""UB input -> local reducer -> one-lane result -> Parallel consumer.

Run on an Ascend device with a matching PTOAS installation. The input is
padded for full-width UB loads; only the logical elements participate.
"""

import tilelang
import tilelang.ascend.language as T


def reducer_scale(n=257, lanes=128, dtype="float32"):
    padded = (n + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((padded,), dtype), B: T.Tensor((padded,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), dtype)
            b = T.alloc_shared((padded,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_reducer((1,), dtype, op="sum")
                result = T.alloc_local((1,), dtype)
                T.reducer_init(acc)
                for i in T.Parallel(n):
                    T.reducer_update(acc[0], a[i])
                T.finalize_reducer(acc, result)
                # Result stays one-lane VMI. Arithmetic belongs in Parallel;
                # the compiler broadcasts at the multiply's consumption point.
                for i in T.Parallel(n):
                    b[i] = a[i] * (result[0] / T.Cast(dtype, 4096))
            T.copy(b, B)

    return main


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    kernel = tilelang.compile(reducer_scale(), target="pto")
    source = torch.ones(384, dtype=torch.float32).npu()
    output = torch.full((384,), -123., dtype=torch.float32).npu()
    kernel(source, output)
    expected = torch.full((384,), -123., dtype=torch.float32)
    expected[:257] = 257 / 4096
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print("PTO SIMD reducer passed")
