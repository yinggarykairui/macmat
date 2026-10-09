// Correctness tests for every MacMat implementation against the double reference.
// No framework dependency: a tiny CHECK macro and a non-zero exit on failure.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "macmat/matrix.hpp"
#include "macmat/multiply.hpp"

using namespace macmat;

namespace {

int g_failures = 0;
int g_checks = 0;

// Documented tolerance for the bounded random suite: relative Frobenius error.
// Inputs are in [-1, 1]; single-precision accumulation over K <= 1024 terms is
// expected to stay far below this.
constexpr double kRelTol = 1e-4;
// Absolute tolerance used only when the reference is exactly zero.
constexpr double kZeroAbsTol = 1e-6;

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        ++g_checks;                                                                          \
        if (!(cond)) {                                                                       \
            ++g_failures;                                                                    \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
        }                                                                                    \
    } while (0)

#define CHECK_THROWS(expr, ExType)                                                           \
    do {                                                                                     \
        ++g_checks;                                                                          \
        bool thrown = false;                                                                 \
        try { (void)(expr); } catch (const ExType&) { thrown = true; }                       \
        catch (...) {}                                                                       \
        if (!thrown) { ++g_failures; std::printf("  FAIL %s:%d: expected %s from %s\n",     \
                                                   __FILE__, __LINE__, #ExType, #expr); }   \
    } while (0)

void fill_garbage(Matrix& m, unsigned seed) {
    Matrix g = Matrix::random(m.rows(), m.cols(), seed, -1000.0f, 1000.0f);
    m.storage() = g.storage();
}

// Runs `impl` with the given options and checks it against the reference.
void check_impl(Impl impl, const Matrix& A, const Matrix& B, const Matrix& ref,
                const MultiplyOptions& opts, const char* label) {
    Matrix C(A.rows(), B.cols());
    fill_garbage(C, 99);  // output must be overwritten regardless of prior contents
    multiply(impl, A, B, C, opts);
    const ErrorStats e = compare(C, ref);
    bool ok;
    if (e.ref_is_zero) {
        ok = e.max_abs_error <= kZeroAbsTol;
    } else {
        ok = e.rel_frobenius_error <= kRelTol && std::isfinite(e.max_abs_error);
    }
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL %-10s %s %zux%zux%zu tile=%zu threads=%u: max_abs=%.3e rel_frob=%.3e\n",
                    impl_name(impl), label, A.rows(), A.cols(), B.cols(), opts.tile, opts.threads,
                    e.max_abs_error, e.rel_frobenius_error);
    }
}

void check_all_impls(const Matrix& A, const Matrix& B, const char* label,
                     std::size_t tile = 32, unsigned threads = 0) {
    Matrix ref(A.rows(), B.cols());
    reference_multiply(A, B, ref);
    MultiplyOptions opts;
    opts.tile = tile;
    opts.threads = threads;
    for (Impl impl : kAllImpls) check_impl(impl, A, B, ref, opts, label);
}

void test_hand_calculated() {
    std::puts("hand-calculated examples");
    // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
    Matrix A(2, 2), B(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 5; B(0, 1) = 6; B(1, 0) = 7; B(1, 1) = 8;
    for (Impl impl : kAllImpls) {
        Matrix C(2, 2, 123.0f);
        multiply(impl, A, B, C);
        CHECK(C(0, 0) == 19 && C(0, 1) == 22 && C(1, 0) == 43 && C(1, 1) == 50);
    }
    // 1x3 * 3x1 = scalar dot product; 3x1 * 1x3 = outer product.
    Matrix r(1, 3), c(3, 1);
    r(0, 0) = 1; r(0, 1) = 2; r(0, 2) = 3;
    c(0, 0) = 4; c(1, 0) = 5; c(2, 0) = 6;
    for (Impl impl : kAllImpls) {
        Matrix dot(1, 1), outer(3, 3);
        multiply(impl, r, c, dot);
        multiply(impl, c, r, outer);
        CHECK(dot(0, 0) == 32);
        CHECK(outer(0, 0) == 4 && outer(1, 2) == 15 && outer(2, 1) == 12);
    }
    // 1x1 matrices.
    Matrix one(1, 1, 2.5f), two(1, 1, -4.0f);
    for (Impl impl : kAllImpls) {
        Matrix C(1, 1);
        multiply(impl, one, two, C);
        CHECK(C(0, 0) == -10.0f);
    }
}

void test_identity_and_zero() {
    std::puts("identity and zero matrices");
    const Matrix A = Matrix::random(37, 41, 1);
    const Matrix I41 = Matrix::identity(41);
    const Matrix I37 = Matrix::identity(37);
    for (Impl impl : kAllImpls) {
        Matrix C(37, 41, 7.0f);
        multiply(impl, A, I41, C);
        CHECK(compare(C, A).max_abs_error == 0.0);
        multiply(impl, I37, A, C);
        CHECK(compare(C, A).max_abs_error == 0.0);
    }
    const Matrix Z(41, 23);  // zeros
    check_all_impls(A, Z, "A*0");
    check_all_impls(Matrix(5, 37), A, "0*A");
}

void test_negative_values() {
    std::puts("negative values");
    const Matrix A = Matrix::random(16, 20, 2, -5.0f, 0.0f);
    const Matrix B = Matrix::random(20, 12, 3, -3.0f, -1.0f);
    check_all_impls(A, B, "negative");
    const Matrix NegI = Matrix::random(20, 20, 4, -1.0f, -1.0f + 1e-30f);  // all -1
    check_all_impls(A, NegI, "all-minus-one");
}

void test_random_square() {
    std::puts("seeded random squares");
    for (std::size_t n : {1, 2, 3, 8, 16, 31, 32, 33, 64, 100, 128, 257}) {
        const Matrix A = Matrix::random(n, n, 10 + static_cast<unsigned>(n));
        const Matrix B = Matrix::random(n, n, 20 + static_cast<unsigned>(n));
        check_all_impls(A, B, "square");
    }
}

void test_rectangular_and_boundaries() {
    std::puts("rectangular and tile-boundary shapes");
    struct Shape { std::size_t m, k, n; };
    const Shape shapes[] = {{17, 31, 13}, {1, 64, 1}, {64, 1, 64}, {3, 200, 5}, {127, 255, 63},
                            {65, 33, 129}, {128, 16, 300}, {9, 9, 1000}};
    for (const Shape& s : shapes) {
        const Matrix A = Matrix::random(s.m, s.k, 5);
        const Matrix B = Matrix::random(s.k, s.n, 6);
        for (std::size_t tile : {1, 3, 16, 32, 64, 128, 1000}) {
            check_all_impls(A, B, "rect", tile);
        }
    }
}

void test_repeated_calls_same_output() {
    std::puts("repeated calls reuse the same output buffer");
    const Matrix A = Matrix::random(45, 70, 7);
    const Matrix B = Matrix::random(70, 38, 8);
    Matrix ref(45, 38);
    reference_multiply(A, B, ref);
    for (Impl impl : kAllImpls) {
        Matrix C(45, 38);
        for (int rep = 0; rep < 4; ++rep) {
            multiply(impl, A, B, C);  // C is not cleared between calls
            CHECK(compare(C, ref).rel_frobenius_error <= kRelTol);
        }
    }
}

void test_thread_counts() {
    std::puts("thread counts that do not divide the row count");
    const Matrix A = Matrix::random(17, 31, 11);
    const Matrix B = Matrix::random(31, 13, 12);
    Matrix ref(17, 13);
    reference_multiply(A, B, ref);
    for (unsigned threads : {1u, 2u, 3u, 4u, 5u, 7u, 16u, 17u, 64u}) {
        MultiplyOptions opts;
        opts.tile = 8;
        opts.threads = threads;
        check_impl(Impl::Parallel, A, B, ref, opts, "threads");
        check_impl(Impl::MicroParallel, A, B, ref, opts, "threads");
    }
    // More threads than rows: must clamp without touching rows outside [0, M).
    const Matrix A2 = Matrix::random(3, 50, 13);
    const Matrix B2 = Matrix::random(50, 40, 14);
    Matrix ref2(3, 40);
    reference_multiply(A2, B2, ref2);
    MultiplyOptions opts;
    opts.threads = 10;
    check_impl(Impl::Parallel, A2, B2, ref2, opts, "threads>rows");
    check_impl(Impl::MicroParallel, A2, B2, ref2, opts, "threads>rows");
}

void test_inputs_unmodified() {
    std::puts("inputs are read-only");
    const Matrix A = Matrix::random(20, 30, 21);
    const Matrix B = Matrix::random(30, 25, 22);
    const std::vector<float> a_copy = A.storage(), b_copy = B.storage();
    for (Impl impl : kAllImpls) {
        Matrix C(20, 25);
        multiply(impl, A, B, C, MultiplyOptions{16, 3});
        CHECK(A.storage() == a_copy);
        CHECK(B.storage() == b_copy);
    }
}

void test_validation() {
    std::puts("validation and readable errors");
    CHECK_THROWS(Matrix(0, 5), std::invalid_argument);
    CHECK_THROWS(Matrix(5, 0), std::invalid_argument);
    CHECK_THROWS(Matrix(static_cast<std::size_t>(1) << 40, static_cast<std::size_t>(1) << 40),
                 std::length_error);
    CHECK_THROWS(checked_element_count(SIZE_MAX, 2), std::length_error);

    Matrix A(4, 3), B(5, 2), C(4, 2);
    for (Impl impl : kAllImpls) CHECK_THROWS(multiply(impl, A, B, C), std::invalid_argument);
    Matrix B2(3, 2), Cbad(4, 3);
    for (Impl impl : kAllImpls) CHECK_THROWS(multiply(impl, A, B2, Cbad), std::invalid_argument);
    // Overlap: use the same object as input and output.
    Matrix S(4, 4);
    CHECK_THROWS(multiply(Impl::Naive, S, S, S), std::invalid_argument);
    Matrix S2(4, 4);
    CHECK_THROWS(multiply(Impl::Blocked, S, S2, S), std::invalid_argument);
    CHECK_THROWS(multiply(Impl::Blocked, A, B2, C, MultiplyOptions{0, 1}), std::invalid_argument);
    // Error messages are readable.
    try {
        multiply(Impl::Naive, A, B, C);
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        CHECK(msg.find("4x3") != std::string::npos && msg.find("5x2") != std::string::npos);
    }
    CHECK(parse_impl("blocked").has_value() && !parse_impl("nope").has_value());
}

void test_error_metrics() {
    std::puts("error metrics");
    Matrix a(2, 2), b(2, 2);
    a(0, 0) = 3; a(1, 1) = 4;
    b(0, 0) = 3; b(1, 1) = 4.5f;
    const ErrorStats e = compare(a, b);
    CHECK(std::fabs(e.max_abs_error - 0.5) < 1e-12);
    CHECK(std::fabs(e.ref_frobenius_norm - std::sqrt(9.0 + 20.25)) < 1e-12);
    CHECK(std::fabs(e.rel_frobenius_error - 0.5 / std::sqrt(29.25)) < 1e-12);
    Matrix z(2, 2);
    CHECK(compare(a, z).ref_is_zero && compare(a, z).rel_frobenius_error == 0.0);
}

void report_precision_sample() {
    // Prints both error measures for a representative case so the README can cite them.
    const Matrix A = Matrix::random(256, 256, 31);
    const Matrix B = Matrix::random(256, 256, 32);
    Matrix ref(256, 256);
    reference_multiply(A, B, ref);
    std::puts("error summary for 256x256x256 random in [-1,1] (threshold rel_frob <= 1e-4):");
    for (Impl impl : kAllImpls) {
        Matrix C(256, 256);
        multiply(impl, A, B, C);
        const ErrorStats e = compare(C, ref);
        std::printf("  %-10s max_abs=%.3e rel_frob=%.3e\n", impl_name(impl), e.max_abs_error,
                    e.rel_frobenius_error);
    }
}

}  // namespace

int main() {
    test_hand_calculated();
    test_identity_and_zero();
    test_negative_values();
    test_random_square();
    test_rectangular_and_boundaries();
    test_repeated_calls_same_output();
    test_thread_counts();
    test_inputs_unmodified();
    test_validation();
    test_error_metrics();
    report_precision_sample();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
