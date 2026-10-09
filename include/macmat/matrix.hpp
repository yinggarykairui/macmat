#pragma once
// MacMat: a minimal contiguous, row-major float matrix.

#include <cstddef>
#include <string>
#include <vector>

namespace macmat {

// Validates that rows*cols elements fit in memory-representable sizes.
// Throws std::invalid_argument for zero dimensions and std::length_error on overflow.
std::size_t checked_element_count(std::size_t rows, std::size_t cols);

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);              // zero-filled
    Matrix(std::size_t rows, std::size_t cols, float fill);  // filled with `fill`

    static Matrix identity(std::size_t n);
    // Uniform values in [lo, hi) from a seeded engine (deterministic).
    static Matrix random(std::size_t rows, std::size_t cols, unsigned seed,
                         float lo = -1.0f, float hi = 1.0f);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    float& operator()(std::size_t i, std::size_t j) noexcept { return data_[i * cols_ + j]; }
    float operator()(std::size_t i, std::size_t j) const noexcept { return data_[i * cols_ + j]; }

    std::vector<float>& storage() noexcept { return data_; }
    const std::vector<float>& storage() const noexcept { return data_; }

    void fill(float value);
    std::string shape_string() const;  // "MxN"

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;
};

}  // namespace macmat
