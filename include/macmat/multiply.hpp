#pragma once
// MacMat multiplication kernels: C = A x B for row-major float matrices.
//
// Every implementation has identical observable behaviour:
//   * A and B are read-only.
//   * C is fully overwritten regardless of its previous contents.
//   * A, B and C must not overlap (checked by multiply()).
//   * Any positive M, K, N work, including partial tiles.
//   * The raw kernels never allocate. (multiply_parallel creates OS threads,
//     which is the per-call overhead the project intentionally measures.)

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "macmat/matrix.hpp"

namespace macmat {

enum class Impl { Naive, Reordered, Blocked, Parallel, Accelerate, Micro, MicroParallel };

const char* impl_name(Impl impl) noexcept;
std::optional<Impl> parse_impl(std::string_view name) noexcept;
// The first five are the project's required implementations; Micro and
// MicroParallel are an extra experiment (register-blocked NEON micro-kernel).
constexpr Impl kAllImpls[] = {Impl::Naive, Impl::Reordered, Impl::Blocked, Impl::Parallel,
                              Impl::Accelerate, Impl::Micro, Impl::MicroParallel};

struct MultiplyOptions {
    std::size_t tile = 32;  // used by Blocked and Parallel
    unsigned threads = 0;   // used by Parallel; 0 = std::thread::hardware_concurrency()
};

// ---- raw kernels (no validation) ---------------------------------------------
// A is M x K, B is K x N, C is M x N, all row-major with no padding.
void multiply_naive(const float* A, const float* B, float* C,
                    std::size_t M, std::size_t K, std::size_t N) noexcept;
void multiply_reordered(const float* A, const float* B, float* C,
                        std::size_t M, std::size_t K, std::size_t N) noexcept;
void multiply_blocked(const float* A, const float* B, float* C,
                      std::size_t M, std::size_t K, std::size_t N, std::size_t tile) noexcept;
// Computes rows [row_begin, row_end) of C with the blocked algorithm. Used by the
// parallel kernel; exposed for tests.
void multiply_blocked_rows(const float* A, const float* B, float* C,
                           std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                           std::size_t row_begin, std::size_t row_end) noexcept;
// Register-blocked micro-kernel (extra experiment). `tile` is the k-block size.
void multiply_micro(const float* A, const float* B, float* C,
                    std::size_t M, std::size_t K, std::size_t N, std::size_t tile) noexcept;
void multiply_micro_rows(const float* A, const float* B, float* C,
                         std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                         std::size_t row_begin, std::size_t row_end) noexcept;
// Signature shared by the row-range kernels above.
using RowKernel = void (*)(const float*, const float*, float*, std::size_t, std::size_t,
                           std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
// Splits rows of C into contiguous blocks, one per thread, each computed with
// `kernel` (multiply_blocked_rows by default). Threads are created and joined on
// every call. threads == 0 means hardware_concurrency().
void multiply_parallel(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                       unsigned threads, RowKernel kernel = multiply_blocked_rows);
// Apple Accelerate cblas_sgemm. Threading inside Accelerate is not controlled by
// this project unless VECLIB_MAXIMUM_THREADS is set in the environment.
void multiply_accelerate(const float* A, const float* B, float* C,
                         std::size_t M, std::size_t K, std::size_t N) noexcept;

// ---- validated entry point ---------------------------------------------------
// Throws std::invalid_argument with a readable message for shape mismatches,
// empty matrices, overlapping buffers, or a zero tile size.
void validate_multiply_args(const Matrix& A, const Matrix& B, const Matrix& C,
                            const MultiplyOptions& opts);
void multiply(Impl impl, const Matrix& A, const Matrix& B, Matrix& C,
              const MultiplyOptions& opts = {});

// ---- reference and error metrics --------------------------------------------
// Independent reference: accumulates each dot product in double, rounds once.
void reference_multiply(const Matrix& A, const Matrix& B, Matrix& C);

struct ErrorStats {
    double max_abs_error = 0.0;
    double rel_frobenius_error = 0.0;  // ||actual - ref||_F / ||ref||_F ; 0 when ref is all-zero
    double ref_frobenius_norm = 0.0;
    bool ref_is_zero = false;          // when true, judge by max_abs_error alone
};
ErrorStats compare(const Matrix& actual, const Matrix& reference);

unsigned default_thread_count() noexcept;

}  // namespace macmat
