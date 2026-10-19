"""Numerical-equivalence/speed probes, not a synthetic quality dataset."""
from pathlib import Path
import importlib.util
import statistics
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v307", ROOT / "solution/solution_v307.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    torch.manual_seed(307)
    shapes = [(1, 1, 64), (2, 10, 128), (4, 128, 256),
              (16, 128, 256), (16, 512, 256), (16, 1024, 256),
              (8, 128, 512), (8, 1024, 512)]
    for batch, rows, dim in shapes:
        x = torch.randn(batch, rows, dim)
        unit = torch.exp2(torch.randint(-4, 4, x.shape).float())
        h = torch.triu(torch.randn(batch, dim, dim) * 0.04)
        h.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        x[..., :8] = torch.arange(-4, 4).float() * unit[..., :8] * 0.125
        started = time.perf_counter()
        ref = mod._gptq_quantize_batched_torch(x, unit, h)
        torch_s = time.perf_counter() - started
        started = time.perf_counter()
        actual = mod._gptq_quantize_batched_np(x, unit, h)
        numpy_s = time.perf_counter() - started
        assert torch.equal(ref, actual), (batch, rows, dim, (ref != actual).sum().item())
        started = time.perf_counter()
        cols = mod._gptq_quantize_batched_columns(x, unit, h)
        columns_s = time.perf_counter() - started
        assert torch.equal(ref, cols), ("columns", batch, rows, dim)
        timings = []
        for fn in (mod._gptq_quantize_batched_torch, mod._gptq_quantize_batched):
            times = []
            for _ in range(3):
                started = time.perf_counter()
                result = fn(x, unit, h)
                times.append(time.perf_counter() - started)
                assert torch.equal(ref, result)
            timings.append(statistics.median(times))
        print((batch, rows, dim), "torch", round(torch_s, 5),
              "numpy", round(numpy_s, 5), "columns", round(columns_s, 5),
              "np_speedup", round(torch_s / numpy_s, 3),
              "col_speedup", round(torch_s / columns_s, 3),
              "dispatch_median_speedup", round(timings[0] / timings[1], 3), flush=True)
    # Shared Hessian and strided inputs are used by the public function too.
    x = torch.randn(3, 23, 384)[..., ::2]
    unit = torch.ones_like(x)
    h = torch.triu(torch.randn(192, 192) * 0.03)
    h.diagonal().fill_(1.0)
    ref = mod._gptq_quantize_batched_torch(x, unit, h)
    for fn in (mod._gptq_quantize_batched_np, mod._gptq_quantize_batched_columns,
               mod._gptq_quantize_batched):
        assert torch.equal(ref, fn(x, unit, h))
    print("all 9 equivalence probes passed; torch threads", torch.get_num_threads())


if __name__ == "__main__":
    main()
