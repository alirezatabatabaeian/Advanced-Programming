#ifndef AP_HW1_H
#define AP_HW1_H

// define libraries
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// using matrix
using Matrix = std::vector<std::vector<double>>;

// Define Namespace
namespace algebra {
Matrix zeros(size_t n, size_t m);
Matrix ones(size_t n, size_t m);
Matrix random(size_t n, size_t m, double min, double max);
void show(const Matrix& matrix);
Matrix multiply(const Matrix& matrix, double c);
Matrix multiply(const Matrix& matrix1, const Matrix& matrix2);
Matrix sum(const Matrix& matrix, double c);
Matrix sum(const Matrix& matrix1, const Matrix& matrix2);
Matrix transpose(const Matrix& matrix);
Matrix minor(const Matrix& matrix, size_t n, size_t m);
double determinant(const Matrix& matrix);
Matrix inverse(const Matrix& matrix);
Matrix concatenate(const Matrix& matrix1, const Matrix& matrix2, int axis);
Matrix ero_swap(const Matrix& matrix, size_t r1, size_t r2);
Matrix ero_multiply(const Matrix& matrix, size_t r, double c);
Matrix ero_sum(const Matrix& matrix, size_t r1, double c, size_t r2);
Matrix upper_triangular(const Matrix& matrix);
}

#endif // AP_HW1_H
