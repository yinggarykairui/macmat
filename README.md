# MacMat

A small C++17 matrix-multiplication engine (`C = A x B`, row-major `float`) built,
tested, benchmarked and profiled on an Apple M4 MacBook. The question driving it:
**how much performance can be gained by changing memory access, cache use and
parallelism, and where do those changes stop helping?**

Full measurements and analysis: [`docs/performance.md`](docs/performance.md).

## Implementations

| Name | What it does | Single-thread GFLOP/s at 1024^3 |
|---|---|---|
| `naive` | i -> j -> k loops; strided reads of B, serial FP reduction | 2.5 |
| `reordered` | i -> k -> j; contiguous B and C rows, vectorized inner loop | 32.6 |
| `blocked` | i-k-j tiled in all three dimensions, `--tile` edge | 29.2 (tile 256), 17.0 (tile 32) |
| `parallel` | `blocked` with output row blocks on `std::thread`s, created and joined per call | 126 (10 threads) |
| `accelerate` | Apple's `cblas_sgemm` (library reference; runs on the AMX units) | 1714 |
| `micro` | *extra experiment:* 4x16 register-blocked NEON micro-kernel, `--tile` = k-block | 88 to 99 |
| `micro_parallel` | *extra experiment:* `micro` across threads | 256 (10 threads), 327 (8 threads) |

The first five are the project's required implementations. The last two were added
after the tile sweep showed that textbook cache blocking of the i-k-j loop could not
beat the un-blocked kernel on this chip; they test the hypothesis that the limit was
register reuse, not cache reuse (it was).

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # -O3 -DNDEBUG -g
cmake --build build -j
./build/macmat_tests                              # correctness (453+ checks)
./build/macmat info                               # chip, OS, compiler, flags, power mode
```

Requires Apple clang (Command Line Tools), CMake >= 3.20, and Python 3 with
matplotlib for charts. Accelerate is linked as a framework.

Sanitizer builds (used for the correctness suite, never for timing):

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMACMAT_SANITIZE=address && cmake --build build-asan -j && ./build-asan/macmat_tests
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DMACMAT_SANITIZE=thread  && cmake --build build-tsan -j && ./build-tsan/macmat_tests
```

Vectorization diagnostics (`-Rpass=loop-vectorize`, `-Rpass-missed=...`, `-Rpass-analysis=...`):

```sh
cmake -S . -B build-vec -DCMAKE_BUILD_TYPE=Release -DMACMAT_VECTORIZE_REPORT=ON
cmake --build build-vec --target macmat 2>&1 | grep remark     # saved in results/vectorization_report.txt
```

## Benchmark CLI

```sh
./build/macmat bench --impl blocked --m 512 --k 512 --n 512 --tile 32 --samples 15 --output results/blocked.csv
./build/macmat bench --impl parallel --m 1024 --k 1024 --n 1024 --tile 256 --threads 4
./build/macmat suite --repeat 3 --tile 32 --output results/suite.csv      # 9 standard shapes x all impls, rotated order
```

Benchmark rules implemented in `benchmarks/bench.cpp`:

- inputs are allocated and seeded outside timing; the output buffer is allocated once
  and every kernel overwrites it (any zero-fill the algorithm needs is inside timing);
- warm-up calls, then at least 15 timed samples with `std::chrono::steady_clock`;
- tiny operations are batched so each sample lasts >= 5 ms (the `batch` column);
- a checksum of C is consumed outside timing so the work cannot be eliminated;
- the CSV records median, Q1, Q3, IQR, min, max, GFLOP/s = 2MKN / s / 1e9, repeat
  index, position in the rotated order, chip, macOS version, compiler, flags, power
  mode, `VECLIB_MAXIMUM_THREADS` and a timestamp.

## Reproduce everything

```sh
scripts/run_experiments.sh          # ~6 min on an M4; writes results/*.csv and results/environment.txt
scripts/profile.sh                  # Instruments via xctrace when Xcode is present, else /usr/bin/sample
python3 scripts/plot_results.py     # results/charts/*.png
```

Keep the machine otherwise idle: a first attempt with a dozen Safari processes alive
(load average 15) made `naive` 3x slower with a 10x larger IQR; after a restart the
numbers were stable to a few percent across repeats.

## Layout

```
CMakeLists.txt
include/macmat/        matrix.hpp (Matrix type), multiply.hpp (kernels, validation, reference)
src/                   matrix.cpp, multiply.cpp (dispatch, reference, error metrics),
                       kernels.cpp (naive/reordered/blocked/parallel), accelerate.cpp, micro.cpp
tests/                 test_correctness.cpp
benchmarks/            bench.hpp/.cpp (timing harness), main.cpp (CLI)
scripts/               run_experiments.sh, profile.sh, plot_results.py, best_tile.py
results/               CSVs, charts/, profiles/, vectorization_report.txt, environment.txt, run.log
docs/performance.md    report
```

## Correctness

Every implementation is checked against an independent reference that accumulates in
`double`: hand-calculated 2x2 and outer/inner products, identity and zero matrices,
negative inputs, seeded random inputs in [-1, 1], rectangular shapes, 17x31x13 and
127x255x63 tile-boundary shapes with tile sizes from 1 to 1000, repeated calls into a
dirty output buffer, thread counts that do not divide the row count (and exceed it),
read-only inputs, and the validation errors (shape mismatch, overflowing allocation,
overlapping buffers, zero tile). Both maximum absolute error and relative Frobenius
error are recorded; the random suite passes at a documented relative threshold of
1e-4 (measured 2.9e-7 at 256^3), and an all-zero reference is judged by absolute error.
