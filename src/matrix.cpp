#include "macmat/matrix.hpp"

#include <limits>
#include <random>
#include <stdexcept>

namespace macmat {

std::size_t checked_element_count(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive, got " +
                                    std::to_string(rows) + "x" + std::to_string(cols));
    }
    const std::size_t max_elems = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (rows > max_elems / cols) {
        throw std::length_error("Matrix " + std::to_string(rows) + "x" + std::to_string(cols) +
                                " overflows the allocation size");
    }
    const std::size_t elems = rows * cols;
    if (elems > std::vector<float>().max_size()) {
        throw std::length_error("Matrix " + std::to_string(rows) + "x" + std::to_string(cols) +
                                " exceeds std::vector<float>::max_size()");
    }
    return elems;
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), 0.0f) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, float fill)
    : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), fill) {}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0f;
    return m;
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, unsigned seed, float lo, float hi) {
    Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : m.data_) v = dist(rng);
    return m;
}

void Matrix::fill(float value) {
    for (float& v : data_) v = value;
}

std::string Matrix::shape_string() const {
    return std::to_string(rows_) + "x" + std::to_string(cols_);
}

}  // namespace macmat
