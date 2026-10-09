# MacMat about

Goal: This document is created so that we are able to explain every decision, every number and every limitation of this
project without notes. Read it top to bottom once, then drill section 7 out loud.

---

## 1. The 30-second pitch

"I built a C++17 matrix-multiplication engine on my M4 MacBook to measure how much
performance you gain from memory-access order, cache blocking, register blocking and
threading, and where each stops helping. Starting from a naive triple loop at 2.5
GFLOP/s, reordering the loops gave 13x, a register-blocked NEON micro-kernel gave
another 3x to ~99 GFLOP/s single-threaded, and threads took it to ~330 GFLOP/s.
Textbook cache blocking never helped, and I can show why: the inner loop was
load/store-bound, not miss-bound, and the M4's 16 MB L2 already holds a 1024^3
working set. Apple's Accelerate is still 5-6x faster because it runs on the AMX
matrix unit, not the NEON pipes. Everything is tested against a double-precision
reference, sanitizer-clean, benchmarked with medians and IQRs over rotated repeats,
and reproducible with one script."

---

## 2. What the code is

```
include/macmat/matrix.hpp    Matrix: rows, cols, std::vector<float> row-major storage
include/macmat/multiply.hpp  Impl enum, kernels, validate_multiply_args, reference, ErrorStats
src/kernels.cpp              naive, reordered, blocked(_rows), parallel
src/micro.cpp                NEON 4x16 register-blocked micro-kernel (+ portable fallback)
src/accelerate.cpp           cblas_sgemm wrapper
src/multiply.cpp             dispatch, validation, double reference, error metrics
benchmarks/bench.cpp         timing harness; main.cpp = CLI (bench / suite / info)
tests/test_correctness.cpp   635 checks, custom CHECK macro, no framework
scripts/                     run_experiments.sh, profile.sh, plot_results.py
```

`C = A x B`, A is MxK, B is KxN, C is MxN, all contiguous row-major floats.
Element (i, j) lives at `data[i*cols + j]`.

Behaviour contract every kernel obeys: inputs read-only, output fully overwritten,
no overlap (checked via pointer-range comparison), arbitrary positive dims, no heap
allocation inside the kernel (the parallel version creates OS threads on purpose:
that cost is what we measure).

---

## 3. Concepts you must own

### 3.1 Loop order and memory access (the 13x)
- naive `i,j,k`: inner loop reads `B[k*N + j]` with stride N floats. Every k touches a
  new cache line of B; for N=1024 that is a 4 KB stride, so no spatial locality and the
  prefetcher is nearly useless. Also `sum += ...` is a serial floating-point dependency
  chain.
- reordered `i,k,j`: inner loop does `c[j] += a * b[j]`, both unit stride, 16 independent
  lanes per iteration after vectorization. Same number of flops, different order.
- Why it matters: DRAM/L2 bandwidth is consumed in 128-byte cache lines on M4. Stride-N
  access wastes 124 of every 128 bytes fetched.

### 3.2 Vectorization (NEON)
- NEON registers are 128 bits = 4 floats. Clang's loop vectorizer reported
  "width 4, interleave 4" for the `j` loops: 4 vector ops per iteration = 16 floats.
- Clang also *reported* vectorizing the naive k-loop. The disassembly shows what it did:
  `fmul.4s` of four products, then four dependent scalar `fadd`s. Without
  `-ffast-math` the compiler must preserve the exact addition order (IEEE floats are
  not associative), so it keeps an "in-order reduction". Vectorized by definition,
  zero speedup in practice. Lesson: read the diagnostic next to the assembly.
- I deliberately did not use `-ffast-math` so all kernels have identical numerics.

### 3.3 Cache hierarchy on the M4 (numbers from sysctl)
- L1d 128 KB per P-core, L2 16 MB shared by the 4 P-cores, 4 MB on the E-cluster,
  128-byte cache lines, 16 GB RAM.
- A 1024x1024 float matrix is 4 MB; all three are 12 MB -> fits L2.
- 2048^3 is 48 MB -> exceeds L2, which is why I ran it as the cache-stress test.

### 3.4 Cache blocking and why it failed here (the negative result)
- Idea: process tile x tile sub-blocks so the A tile and a strip of B stay in L1 while
  they are reused. Textbook says tile ~ 32-64 for a 128 KB L1.
- Measured: throughput rises monotonically with tile size and plateaus at the
  un-blocked value. No tile beats `reordered` at any shape, even at 2048^3.
- Two pieces of evidence that it is *not* a cache effect:
  1. Tile 16 is equally slow (~16 GFLOP/s) at 32^3 (everything in L1) and at 1024^3.
     A cache effect scales with problem size; this does not.
  2. Working set fits L2 for all standard shapes, and the prefetcher handles
     sequential 4 KB rows of B.
- The real cost: a 16-wide j range runs the 16-float vector body once per k and then
  pays loop prologue/epilogue; and `c[j] += a*b[j]` loads and stores C on every k, so
  successive k steps are chained through memory on the same addresses
  (store-to-load forwarding latency, ~several cycles, instead of back-to-back FMAs).
- Only hint of a cache benefit in the whole project: at 2048^3 with the micro-kernel,
  k-block 256 beat k-block 1024 by 5%.

### 3.5 Register blocking (the 3x)
- Arithmetic intensity of the inner loop: reordered does per element 1 load of B,
  1 load of C, 1 store of C, 1 FMA -> load/store-bound.
- Micro-kernel: hold a 4-row x 16-column block of C in 16 NEON registers. Per k step:
  load 4 vectors of B (16 floats), broadcast 4 scalars of A, do 16 vector FMAs. C is
  loaded/stored once per k-block, not once per k. Now FMA-bound.
- Peak estimate for one P-core: 4 FMA pipes x 4 lanes x 2 flop x ~4.4 GHz = ~140
  GFLOP/s. `micro` hits ~99, roughly 70% of peak.
- Edge handling: rows not a multiple of 4 or columns not a multiple of 16 take a scalar
  path. That is why 127x255x63 only gets 24 GFLOP/s (N=63 = three full 16-wide tiles
  + 15 scalar columns). Fix would be padding or a narrower edge kernel.
- Written with `arm_neon.h` intrinsics (`vld1q_f32`, `vdupq_n_f32`, `vfmaq_f32`,
  `vst1q_f32`), with a plain-array fallback for Intel.

### 3.6 Threads
- `std::thread`, one contiguous block of output rows per thread. Rows are disjoint so
  no synchronization is needed; inputs are read-only and shared.
- Threads created and joined on every call (by design, to measure that cost). A
  fixed-size `std::thread[256]` array avoids a heap allocation; the calling thread
  computes block 0 itself; threads are clamped to min(threads, M).
- Uneven splits: first `M % threads` blocks get one extra row.
- Measured: thread creation+join of 9 threads costs ~45 us (parallel at 32^3 = 53 us
  vs 4 us single-threaded). Below ~128^3 threading loses.
- Scaling: ~3-3.7x at 4 threads (4 P-cores), 4.3-4.6x at 10 for the blocked kernel.
  `micro_parallel` peaks at 8 threads and is *slower* at 10: the static split gives
  E-cores the same number of rows as P-cores, so the slowest thread sets the finish
  time, and the faster kernel makes the fixed overhead proportionally larger.
- Profiles: ~94% of samples in the kernel, ~5% main thread blocked in `join`
  (`__ulock_wait`), <1-5% thread start-up.
- Next steps I would name: persistent worker pool, dynamic/work-stealing row chunks,
  pinning or sizing work by core type.

### 3.7 Accelerate and AMX
- `cblas_sgemm(CblasRowMajor, NoTrans, NoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N)`.
  beta = 0 means C is overwritten without being read, matching our contract.
- 1.5-1.8 TFLOP/s. That is 10x above NEON peak of all four P-cores combined, so it
  cannot be NEON: Apple routes sgemm to the AMX matrix coprocessor (undocumented ISA,
  no public API). So the gap is largely a hardware-unit comparison, not a loop one.
- Threading uncontrolled by default. `VECLIB_MAXIMUM_THREADS=1` reduced 1024^3 from
  1827 to 1576 GFLOP/s, so it is mostly single-threaded at these sizes.
- Also near-zero call overhead: 0.2 us at 32^3.

### 3.8 Benchmark methodology (expect to be grilled on this)
- `std::chrono::steady_clock` (monotonic; `system_clock` can jump).
- Inputs seeded and allocated outside timing; output allocated once; any zeroing the
  algorithm needs is inside timing.
- 3 warm-ups (page faults, cache fill, Accelerate's internal init), then 15 samples.
- Batching: if one call is shorter than 5 ms, run `batch` calls per sample so the
  ~20-40 ns timer cost is negligible; report per-call time.
- Dead-code elimination: a checksum of C is accumulated and printed; otherwise the
  compiler could legally drop the work.
- Report median (robust to outliers), Q1/Q3/IQR (spread), min/max, and
  GFLOP/s = 2MKN / seconds / 1e9 (2 because each term is one multiply + one add).
- Rotate implementation order between repeats so thermal ramp or frequency state
  does not systematically favour whoever runs first; repeats agreed within a few %.
- Record chip, macOS, compiler, flags, threads, tile, power mode, timestamp in every
  CSV row so results are self-describing.
- Same flags for every implementation: `-O3 -DNDEBUG -g` (`-g` keeps symbols for the
  profiler without changing codegen).
- Real-world lesson: the first run with Safari open (load average 15) was 3x slower
  and 10x noisier. Benchmark on an idle machine.

### 3.9 Correctness methodology
- Independent reference accumulates in `double`, rounds once to float.
- Metrics: max absolute error and relative Frobenius error
  ||C - R||_F / ||R||_F. Threshold 1e-4 for random inputs in [-1,1]; measured 2.9e-7
  at 256^3. If the reference is all zero, judge by absolute error (relative is
  undefined).
- Why float results differ from the reference at all: different summation order
  (vectorized lanes, k-blocking) rounds differently; error grows ~sqrt(K) * eps.
- Test matrix: hand-computed 2x2, dot/outer products, 1x1; identity (exact);
  zeros; negatives; seeded random squares 1..257; rectangular and tile-boundary
  shapes (17x31x13, 127x255x63) with tiles 1..1000; repeated calls into a dirty
  output buffer; thread counts that do not divide or exceed M; inputs unchanged;
  validation errors (shape mismatch, overflow, overlap, zero tile) with readable messages.
- Overflow check: `rows > SIZE_MAX/sizeof(float)/cols` before allocating.

### 3.10 Sanitizers and profiling
- ASan + UBSan: memory errors (out-of-bounds, use-after-free) and undefined behaviour
  (signed overflow, misaligned access). TSan: data races; run separately because TSan
  is incompatible with ASan. All on Debug builds; never time sanitized builds.
- Profiling: Instruments Time Profiler needs full Xcode, which this Mac lacks (CLT
  only). Used `/usr/bin/sample`, also a 1 ms call-stack sampler. Neither measures cache
  misses; all cache claims are inferred from timing across sizes. Be upfront about that.

---

## 4. Numbers to memorize (1024^3 unless noted)

| | GFLOP/s | latency |
|---|---|---|
| naive | 2.5 | 870 ms |
| reordered | 32.6 | 66 ms |
| blocked, tile 32 / 256 | 17 / 29 | 73 ms (256) |
| parallel, 10 thr | 126 | 17 ms |
| micro | 99 | 22 ms |
| micro_parallel, 8 thr / 10 thr | 327 / 256 | 6.6 / 8.5 ms |
| accelerate | 1714 | 1.25 ms |

Other: thread creation ~45 us; parallel loses below 128^3; M4 = 4P + 6E, 128 KB L1d,
16 MB L2; 1024^2 floats = 4 MB; NEON vector = 4 floats; micro-tile 4x16 = 16 regs.

---

## 5. Honest answers to "what are the weaknesses?"

- Matrix multiplication is a classic exercise; the value is in the measurement
  discipline and the explained negative result, not novelty.
- No Instruments traces (no Xcode on the machine); profiles are from `sample`.
- No hardware counters, so cache behaviour is inferred, not measured.
- Micro-kernel edges are scalar (127x255x63 is 4x slower than square shapes).
- Static row split and per-call thread creation limit scaling; no worker pool yet.
- No packing of A/B panels, which is the next standard GEMM step.
- Single machine, single compiler version; not cross-validated on Intel.

## 6. Resume bullets

- Built a C++17/CMake matrix-multiplication engine with five kernels (naive, reordered,
  cache-blocked, std::thread parallel, Accelerate) plus a NEON register-blocked
  micro-kernel; 2.5 -> 327 GFLOP/s on Apple M4.
- Designed a benchmark harness (steady_clock, warm-up, batching, median/IQR, rotated
  repeats, DCE guard) and ran tile, thread and loop-order experiments with predictions
  recorded before measurement.
- Showed with controlled experiments that cache blocking cannot help below the 16 MB
  L2 and that the kernel was load/store-bound; fixed it with register blocking (3x).
- 635 correctness checks against a double-precision reference; ASan/UBSan/TSan clean.

---

## 7. Interview drill (say the answer out loud)

1. **Why does i-k-j beat i-j-k?** Unit-stride B and C, vectorizable, prefetch-friendly;
   i-j-k strides B by N and serializes the sum.
2. **Clang said it vectorized naive. Why no speedup?** In-order FP reduction: vector
   multiply then four dependent scalar adds, plus a strided gather. No fast-math.
3. **Why didn't blocking help?** Tile-size penalty is size-independent (even at 32^3),
   so it is loop overhead and the C load/store chain, not misses; and the working set
   fits the 16 MB L2 anyway. Only at 2048^3 with the micro-kernel did k-blocking give 5%.
4. **What is the arithmetic intensity change in the micro-kernel?** From ~1 FMA per
   3 memory ops to 16 FMAs per 4 vector loads + 4 broadcasts; C touched once per
   k-block.
5. **Why 4x16?** 16 accumulators + 4 B vectors + a few A broadcasts fit in 32 NEON
   registers; 16 columns = 4 vectors matches 4 FMA pipes.
6. **Why does micro_parallel get slower at 10 threads?** Static equal row split across
   4 fast and 6 slow cores; the slowest E-core thread sets the finish time; thread
   creation is fixed cost.
7. **How did you avoid the compiler deleting the benchmark?** Checksum of C read after
   timing and printed.
8. **Why median and IQR, not mean and stddev?** Robust to scheduler/thermal outliers;
   latency distributions are skewed.
9. **How do you know results are stable?** Three repeats with rotated order, spread of
   a few percent; CSV stores Q1/Q3; noisy first run under load was discarded and
   explained.
10. **Why is Accelerate 6x faster than your best?** AMX coprocessor, above NEON peak;
    plus packing and tuned kernels; its threading was uncontrolled, and single-thread
    env only cost ~15%.
11. **How did you test thread safety?** Disjoint output rows, read-only inputs; TSan
    build of the full test suite including non-dividing thread counts.
12. **What error metric and why?** Max abs for worst element; relative Frobenius for
    overall; threshold 1e-4, measured 3e-7; zero-reference handled separately.
13. **What would you do next?** Pack panels, 8x12 or 6x16 micro-tile, worker pool
    with dynamic chunks, edge-kernel for non-multiple shapes, then a CUDA/Metal port
    of the same experiment to compare GPU tiling with shared memory and register
    blocking.
14. **How does this relate to GPUs / NVIDIA?** Same principles: coalesced access (loop
    order), shared-memory tiling (cache blocking), register tiling (micro-kernel),
    occupancy vs. per-thread resources (threads vs. overhead), and library baselines
    (cuBLAS plays Accelerate's role, with tensor cores playing AMX's).
