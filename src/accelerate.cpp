#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "macmat/multiply.hpp"

namespace macmat {

void multiply_accelerate(const float* A, const float* B, float* C,
                         std::size_t M, std::size_t K, std::size_t N) noexcept {
    // Row-major: C[MxN] = 1.0 * A[MxK] * B[KxN] + 0.0 * C. beta = 0 overwrites C
    // without reading it (BLAS semantics), matching the other kernels.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, A, static_cast<int>(K), B, static_cast<int>(N),
                0.0f, C, static_cast<int>(N));
}

}  // namespace macmat
