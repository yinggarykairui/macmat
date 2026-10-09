// Extra experiment (beyond the four required kernels): register-blocked
// micro-kernel. A 4-row x 16-column block of C is kept in NEON registers while
// k runs over a cache block of size `tile`, so C is read and written once per
// k-block instead of once per k. B is streamed as four 128-bit vectors per k and
// A as four scalar broadcasts.
#include "macmat/multiply.hpp"

#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace macmat {

namespace {
constexpr std::size_t MR = 4;   // rows of C per micro-tile
constexpr std::size_t NR = 16;  // columns of C per micro-tile (4 NEON vectors)

// Generic edge path for partial micro-tiles (mr <= 4 rows, nr <= 16 columns).
inline void edge_tile(const float* __restrict A, const float* __restrict B, float* __restrict C,
                      std::size_t K, std::size_t N, std::size_t i, std::size_t j,
                      std::size_t mr, std::size_t nr, std::size_t k0, std::size_t k1) noexcept {
    for (std::size_t r = 0; r < mr; ++r) {
        float* __restrict c = C + (i + r) * N + j;
        const float* __restrict a = A + (i + r) * K;
        for (std::size_t k = k0; k < k1; ++k) {
            const float aik = a[k];
            const float* __restrict b = B + k * N + j;
            for (std::size_t cc = 0; cc < nr; ++cc) c[cc] += aik * b[cc];
        }
    }
}

#if defined(__ARM_NEON)
inline void full_tile(const float* __restrict A, const float* __restrict B, float* __restrict C,
                      std::size_t K, std::size_t N, std::size_t i, std::size_t j,
                      std::size_t k0, std::size_t k1) noexcept {
    float32x4_t acc[MR][4];
    for (std::size_t r = 0; r < MR; ++r)
        for (std::size_t v = 0; v < 4; ++v) acc[r][v] = vld1q_f32(C + (i + r) * N + j + 4 * v);
    const float* a0 = A + (i + 0) * K;
    const float* a1 = A + (i + 1) * K;
    const float* a2 = A + (i + 2) * K;
    const float* a3 = A + (i + 3) * K;
    for (std::size_t k = k0; k < k1; ++k) {
        const float* b = B + k * N + j;
        const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8), b3 = vld1q_f32(b + 12);
        float32x4_t a;
        a = vdupq_n_f32(a0[k]);
        acc[0][0] = vfmaq_f32(acc[0][0], a, b0); acc[0][1] = vfmaq_f32(acc[0][1], a, b1);
        acc[0][2] = vfmaq_f32(acc[0][2], a, b2); acc[0][3] = vfmaq_f32(acc[0][3], a, b3);
        a = vdupq_n_f32(a1[k]);
        acc[1][0] = vfmaq_f32(acc[1][0], a, b0); acc[1][1] = vfmaq_f32(acc[1][1], a, b1);
        acc[1][2] = vfmaq_f32(acc[1][2], a, b2); acc[1][3] = vfmaq_f32(acc[1][3], a, b3);
        a = vdupq_n_f32(a2[k]);
        acc[2][0] = vfmaq_f32(acc[2][0], a, b0); acc[2][1] = vfmaq_f32(acc[2][1], a, b1);
        acc[2][2] = vfmaq_f32(acc[2][2], a, b2); acc[2][3] = vfmaq_f32(acc[2][3], a, b3);
        a = vdupq_n_f32(a3[k]);
        acc[3][0] = vfmaq_f32(acc[3][0], a, b0); acc[3][1] = vfmaq_f32(acc[3][1], a, b1);
        acc[3][2] = vfmaq_f32(acc[3][2], a, b2); acc[3][3] = vfmaq_f32(acc[3][3], a, b3);
    }
    for (std::size_t r = 0; r < MR; ++r)
        for (std::size_t v = 0; v < 4; ++v) vst1q_f32(C + (i + r) * N + j + 4 * v, acc[r][v]);
}
#else
// Portable fallback (Intel Macs): same register-tile shape expressed with
// fixed-size arrays that the compiler can keep in vector registers.
inline void full_tile(const float* __restrict A, const float* __restrict B, float* __restrict C,
                      std::size_t K, std::size_t N, std::size_t i, std::size_t j,
                      std::size_t k0, std::size_t k1) noexcept {
    float acc[MR][NR];
    for (std::size_t r = 0; r < MR; ++r)
        for (std::size_t cc = 0; cc < NR; ++cc) acc[r][cc] = C[(i + r) * N + j + cc];
    for (std::size_t k = k0; k < k1; ++k) {
        const float* b = B + k * N + j;
        for (std::size_t r = 0; r < MR; ++r) {
            const float a = A[(i + r) * K + k];
            for (std::size_t cc = 0; cc < NR; ++cc) acc[r][cc] += a * b[cc];
        }
    }
    for (std::size_t r = 0; r < MR; ++r)
        for (std::size_t cc = 0; cc < NR; ++cc) C[(i + r) * N + j + cc] = acc[r][cc];
}
#endif
}  // namespace

void multiply_micro_rows(const float* A, const float* B, float* C,
                         std::size_t M, std::size_t K, std::size_t N, std::size_t tile,
                         std::size_t row_begin, std::size_t row_end) noexcept {
    (void)M;
    if (tile == 0) tile = 1;
    for (std::size_t i = row_begin; i < row_end; ++i) {
        float* c = C + i * N;
        for (std::size_t j = 0; j < N; ++j) c[j] = 0.0f;
    }
    for (std::size_t kk = 0; kk < K; kk += tile) {
        const std::size_t k_end = std::min(kk + tile, K);
        for (std::size_t i = row_begin; i < row_end; i += MR) {
            const std::size_t mr = std::min(MR, row_end - i);
            for (std::size_t j = 0; j < N; j += NR) {
                const std::size_t nr = std::min(NR, N - j);
                if (mr == MR && nr == NR) {
                    full_tile(A, B, C, K, N, i, j, kk, k_end);
                } else {
                    edge_tile(A, B, C, K, N, i, j, mr, nr, kk, k_end);
                }
            }
        }
    }
}

void multiply_micro(const float* A, const float* B, float* C,
                    std::size_t M, std::size_t K, std::size_t N, std::size_t tile) noexcept {
    multiply_micro_rows(A, B, C, M, K, N, tile, 0, M);
}

}  // namespace macmat
