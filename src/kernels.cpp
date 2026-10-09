// Scalar / compiler-vectorized kernels: naive, reordered, blocked, parallel.
#include "macmat/multiply.hpp"

#include <algorithm>
#include <cstring>
#include <thread>

namespace macmat {

// i -> j -> k. The inner loop walks B down a column (stride N floats) and
// accumulates into a scalar: a dependent reduction the compiler will not
// vectorize without -ffast-math.
void multiply_naive(const float* __restrict A, const float* __restrict B, float* __restrict C,
                    std::size_t M, std::size_t K, std::size_t N) noexcept {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// i -> k -> j. The inner loop streams contiguous rows of B and C, so it is
// unit-stride on both and has no loop-carried dependency: vectorizable.
void multiply_reordered(const float* __restrict A, const float* __restrict B, float* __restrict C,
                        std::size_t M, std::size_t K, std::size_t N) noexcept {
    for (std::size_t i = 0; i < M; ++i) {
        float* __restrict c = C + i * N;
        for (std::size_t j = 0; j < N; ++j) c[j] = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            const float a = A[i * K + k];
            const float* __restrict b = B + k * N;
            for (std::size_t j = 0; j < N; ++j) {
                c[j] += a * b[j];
            }
        }
    }
}

// Tiled i -> k -> j. For each (row tile, k tile) the A sub-block (tile x tile)
// stays hot while a tile-wide strip of B and C is streamed. Partial tiles at
// the matrix edges are handled with min().
void multiply_blocked_rows(const float* __restrict A, const float* __restrict B, float* __restrict C,
                           std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                           std::size_t row_begin, std::size_t row_end) noexcept {
    (void)M;
    if (tile == 0) tile = 1;
    for (std::size_t i = row_begin; i < row_end; ++i) {
        float* c = C + i * N;
        for (std::size_t j = 0; j < N; ++j) c[j] = 0.0f;
    }
    for (std::size_t ii = row_begin; ii < row_end; ii += tile) {
        const std::size_t i_end = std::min(ii + tile, row_end);
        for (std::size_t kk = 0; kk < K; kk += tile) {
            const std::size_t k_end = std::min(kk + tile, K);
            for (std::size_t jj = 0; jj < N; jj += tile) {
                const std::size_t j_end = std::min(jj + tile, N);
                for (std::size_t i = ii; i < i_end; ++i) {
                    float* __restrict c = C + i * N;
                    const float* __restrict a = A + i * K;
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const float aik = a[k];
                        const float* __restrict b = B + k * N;
                        for (std::size_t j = jj; j < j_end; ++j) {
                            c[j] += aik * b[j];
                        }
                    }
                }
            }
        }
    }
}

void multiply_blocked(const float* A, const float* B, float* C,
                      std::size_t M, std::size_t K, std::size_t N, std::size_t tile) noexcept {
    multiply_blocked_rows(A, B, C, M, K, N, tile, 0, M);
}

unsigned default_thread_count() noexcept {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1u : hc;
}

namespace {
// Upper bound on worker threads so the dispatcher needs no heap container.
constexpr unsigned kMaxThreads = 256;
}  // namespace

void multiply_parallel(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                       unsigned threads, RowKernel kernel) {
    if (threads == 0) threads = default_thread_count();
    threads = std::min<unsigned>(threads, kMaxThreads);
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, M));  // never more threads than rows
    if (threads <= 1) {
        kernel(A, B, C, M, K, N, tile, 0, M);
        return;
    }
    // Rows are split into `threads` contiguous blocks; the first `rem` blocks get
    // one extra row so counts that do not divide M are handled evenly.
    const std::size_t base = M / threads;
    const std::size_t rem = M % threads;
    auto block_bounds = [&](unsigned t, std::size_t& begin, std::size_t& end) {
        begin = t * base + std::min<std::size_t>(t, rem);
        end = begin + base + (t < rem ? 1 : 0);
    };

    std::thread workers[kMaxThreads];  // default-constructed, not joinable, no heap allocation
    for (unsigned t = 1; t < threads; ++t) {
        std::size_t begin, end;
        block_bounds(t, begin, end);
        workers[t] = std::thread(kernel, A, B, C, M, K, N, tile, begin, end);
    }
    // The calling thread computes block 0 itself.
    {
        std::size_t begin, end;
        block_bounds(0, begin, end);
        kernel(A, B, C, M, K, N, tile, begin, end);
    }
    for (unsigned t = 1; t < threads; ++t) workers[t].join();
}

}  // namespace macmat
