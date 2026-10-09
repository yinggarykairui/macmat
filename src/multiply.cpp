// Validation, dispatch, reference implementation and error metrics.
#include "macmat/multiply.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace macmat {

const char* impl_name(Impl impl) noexcept {
    switch (impl) {
        case Impl::Naive: return "naive";
        case Impl::Reordered: return "reordered";
        case Impl::Blocked: return "blocked";
        case Impl::Parallel: return "parallel";
        case Impl::Accelerate: return "accelerate";
        case Impl::Micro: return "micro";
        case Impl::MicroParallel: return "micro_parallel";
    }
    return "unknown";
}

std::optional<Impl> parse_impl(std::string_view name) noexcept {
    for (Impl impl : kAllImpls) {
        if (name == impl_name(impl)) return impl;
    }
    return std::nullopt;
}

namespace {

bool ranges_overlap(const float* a, std::size_t na, const float* b, std::size_t nb) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b);
    const auto a1 = a0 + na * sizeof(float);
    const auto b1 = b0 + nb * sizeof(float);
    return a0 < b1 && b0 < a1;
}

}  // namespace

void validate_multiply_args(const Matrix& A, const Matrix& B, const Matrix& C,
                            const MultiplyOptions& opts) {
    if (A.empty() || B.empty() || C.empty()) {
        throw std::invalid_argument("multiply: all matrices must be non-empty");
    }
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("multiply: inner dimensions differ: A is " + A.shape_string() +
                                    " but B is " + B.shape_string() + " (A.cols must equal B.rows)");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument("multiply: output C is " + C.shape_string() + " but A x B is " +
                                    std::to_string(A.rows()) + "x" + std::to_string(B.cols()));
    }
    if (ranges_overlap(C.data(), C.size(), A.data(), A.size()) ||
        ranges_overlap(C.data(), C.size(), B.data(), B.size())) {
        throw std::invalid_argument("multiply: output C must not overlap inputs A or B");
    }
    if (opts.tile == 0) {
        throw std::invalid_argument("multiply: tile size must be positive");
    }
}

void multiply(Impl impl, const Matrix& A, const Matrix& B, Matrix& C, const MultiplyOptions& opts) {
    validate_multiply_args(A, B, C, opts);
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    switch (impl) {
        case Impl::Naive: multiply_naive(A.data(), B.data(), C.data(), M, K, N); break;
        case Impl::Reordered: multiply_reordered(A.data(), B.data(), C.data(), M, K, N); break;
        case Impl::Blocked: multiply_blocked(A.data(), B.data(), C.data(), M, K, N, opts.tile); break;
        case Impl::Parallel:
            multiply_parallel(A.data(), B.data(), C.data(), M, K, N, opts.tile, opts.threads);
            break;
        case Impl::Accelerate: multiply_accelerate(A.data(), B.data(), C.data(), M, K, N); break;
        case Impl::Micro: multiply_micro(A.data(), B.data(), C.data(), M, K, N, opts.tile); break;
        case Impl::MicroParallel:
            multiply_parallel(A.data(), B.data(), C.data(), M, K, N, opts.tile, opts.threads,
                              multiply_micro_rows);
            break;
    }
}

void reference_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    validate_multiply_args(A, B, C, MultiplyOptions{});
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += static_cast<double>(A(i, k)) * static_cast<double>(B(k, j));
            }
            C(i, j) = static_cast<float>(sum);
        }
    }
}

ErrorStats compare(const Matrix& actual, const Matrix& reference) {
    if (actual.rows() != reference.rows() || actual.cols() != reference.cols()) {
        throw std::invalid_argument("compare: shapes differ: " + actual.shape_string() + " vs " +
                                    reference.shape_string());
    }
    ErrorStats s;
    double diff_sq = 0.0, ref_sq = 0.0;
    const float* a = actual.data();
    const float* r = reference.data();
    for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        const double d = static_cast<double>(a[idx]) - static_cast<double>(r[idx]);
        s.max_abs_error = std::max(s.max_abs_error, std::fabs(d));
        diff_sq += d * d;
        ref_sq += static_cast<double>(r[idx]) * static_cast<double>(r[idx]);
    }
    s.ref_frobenius_norm = std::sqrt(ref_sq);
    s.ref_is_zero = (ref_sq == 0.0);
    s.rel_frobenius_error = s.ref_is_zero ? 0.0 : std::sqrt(diff_sq) / s.ref_frobenius_norm;
    return s;
}

}  // namespace macmat
