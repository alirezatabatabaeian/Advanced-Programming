#include "hw1.h"

Matrix algebra::zeros(size_t n, size_t m) // Matrix with zero elements
{
    if (n > 0 && m > 0) {
        Matrix Output_Matrix; // define output
        std::vector<double> Rows(m); // define rows

        for (size_t i {}; i < n; i++) {
            Output_Matrix.push_back(Rows); // add rows to vector
        }

        return Output_Matrix; // return output matrix
    } else {
        // logical error
        throw std::logic_error("size should be positive integers");
    }
}

Matrix algebra::ones(size_t n, size_t m) // Matrix with one elements
{
    if (n > 0 && m > 0) {
        Matrix Output_Matrix; // define output
        std::vector<double> Rows(m); // define rows

        for (size_t i {}; i < m; i++) {
            Rows[i] = 1; // define rows
        }

        for (size_t i {}; i < n; i++) {
            Output_Matrix.push_back(Rows); // add rows to vector
        }

        return Output_Matrix; // return output matrix
    } else {
        // logical error
        throw std::logic_error("size should be positive integers");
    }
}

// Matrix with random elements
Matrix algebra::random(size_t n, size_t m, double min, double max)
{
    if (n > 0 && m > 0) {
        std::random_device random_device; // define random generator
        std::mt19937 engine { random_device() };
        std::uniform_real_distribution<double> dist(min, max);

        Matrix Output_Matrix; // define output
        std::vector<double> Rows(m); // define rows
        if (min > max) {
            // logical error
            throw std::logic_error("min cannot be greater than max");
        } else {
            for (size_t i {}; i < n; i++) {
                for (size_t j {}; j < m; j++) {
                    Rows[j] = dist(engine); // generate random numbers
                }
                Output_Matrix.push_back(Rows); // add rows to vector
            }
        }

        return Output_Matrix; // return output matrix
    } else {
        // logical error
        throw std::logic_error("size should be positive integers");
    }
}

// show a Matrix
void algebra::show(const Matrix& matrix)
{
    size_t n { matrix.size() }; // define size of matrix
    size_t m { matrix[0].size() };

    for (size_t i {}; i < n; i++) {
        for (size_t j {}; j < m; j++) {
            // define precision && width
            std::cout << std::setw(7) << std::fixed
                      << std::setprecision(3) << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

// multiply a constant to a Matrix
Matrix algebra::multiply(const Matrix& matrix, double c)
{
    size_t n { matrix.size() }; // define size of matrix
    size_t m { matrix[0].size() };
    Matrix Output_Matrix { algebra::zeros(n, m) };

    for (size_t i {}; i < n; i++) {
        for (size_t j {}; j < m; j++) {
            Output_Matrix[i][j] = c * matrix[i][j]; // multiply a number
        }
    }

    return Output_Matrix; // return output matrix
}

// multiply a Matrix to a Matrix
Matrix algebra::multiply(const Matrix& matrix1, const Matrix& matrix2)
{
    if (matrix1.empty() && matrix2.empty()) {
        Matrix Output_Matrix {}; // empty matrix
        return Output_Matrix; // return output matrix
    } else {
        size_t n1 { matrix1.size() }; // define size of matrix
        size_t m1 { matrix1[0].size() };
        size_t n2 { matrix2.size() };
        size_t m2 { matrix2[0].size() };
        if (m1 != n2) {
            // logical error
            throw std::logic_error(
                "matrices with wrong dimensions cannot be multiplied");
        } else {
            // define size of matrix
            Matrix Output_Matrix { algebra::zeros(n1, m2) };
            for (size_t i {}; i < n1; i++) {
                for (size_t j {}; j < m2; j++) {
                    for (size_t k {}; k < m1; k++) {
                        // multiply
                        Output_Matrix[i][j] += (matrix1[i][k]) * (matrix2[k][j]);
                    }
                }
            }
            return Output_Matrix; // return output matrix
        }
    }
}

// sum a constant with a Matrix
Matrix algebra::sum(const Matrix& matrix, double c)
{
    if (matrix.empty()) {
        Matrix Output_Matrix {}; // empty matrix
        return Output_Matrix; // return output matrix
    } else {
        size_t n { matrix.size() }; // define size of matrix
        size_t m { matrix[0].size() };
        Matrix Output_Matrix { algebra::zeros(n, m) }; // define size of matrix
        for (size_t i {}; i < n; i++) {
            for (size_t j {}; j < m; j++) {
                Output_Matrix[i][j] = matrix[i][j] + c; // sum
            }
        }
        return Output_Matrix; // return output matrix
    }
}

// sum a Matrix with a Matrix
Matrix algebra::sum(const Matrix& matrix1, const Matrix& matrix2)
{
    if (matrix1.empty() && matrix2.empty()) {
        Matrix Output_Matrix {}; // empty matrix
        return Output_Matrix; // return output matrix
    } else if (matrix1.empty() || matrix2.empty()) {
        // logical error
        throw std::logic_error(
            "matrices with wrong dimensions cannot be summed");
    } else {
        size_t n1 { matrix1.size() }; // define size of matrix
        size_t m1 { matrix1[0].size() };
        size_t n2 { matrix2.size() };
        size_t m2 { matrix2[0].size() };
        if ((n1 != n2) || (m1 != m2)) {
            // logical error
            throw std::logic_error(
                "matrices with wrong dimensions cannot be summed");
        } else {
            // define size of matrix
            Matrix Output_Matrix { algebra::zeros(n1, m1) };
            for (size_t i {}; i < n1; i++) {
                for (size_t j {}; j < m1; j++) {
                    Output_Matrix[i][j] = matrix1[i][j] + matrix2[i][j]; // sum
                }
            }
            return Output_Matrix; // return output matrix
        }
    }
}

// transpose a Matrix
Matrix algebra::transpose(const Matrix& matrix)
{
    if (matrix.empty()) {
        Matrix Output_Matrix {}; // empty matrix
        return Output_Matrix; // return output matrix
    } else {
        size_t n { matrix.size() }; // define size of matrix
        size_t m { matrix[0].size() }; // define size of matrix
        Matrix Output_Matrix { algebra::zeros(m, n) }; // define size of matrix
        for (size_t i {}; i < n; i++) {
            for (size_t j {}; j < m; j++) {
                Output_Matrix[j][i] = matrix[i][j]; // copy
            }
        }
        return Output_Matrix; // return output matrix
    }
}

// minor of a Matrix
Matrix algebra::minor(const Matrix& matrix, size_t n, size_t m)
{
    size_t r { matrix.size() }; // define size of matrix
    size_t c { matrix[0].size() };
    Matrix Output_Matrix { algebra::zeros(r - 1, c - 1) }; // define matrix
    if ((n >= 0 && n < r) && (m >= 0 && m < c)) {
        // remove row and column
        for (size_t i {}; i < r; i++) {
            for (size_t j {}; j < c; j++) {
                if (i < n && j < m) {
                    Output_Matrix[i][j] = matrix[i][j];
                } else if (i < n && j > m) {
                    Output_Matrix[i][j - 1] = matrix[i][j];
                } else if (i > n && j < m) {
                    Output_Matrix[i - 1][j] = matrix[i][j];
                } else if (i > n && j > m) {
                    Output_Matrix[i - 1][j - 1] = matrix[i][j];
                }
            }
        }

        return Output_Matrix;
    } else {
        // logical error
        throw std::logic_error("row & column should be consistent");
    }
}

// determinant of a Matrix
double algebra::determinant(const Matrix& matrix)
{
    if (matrix.empty()) {
        return 1.0;
    } else {
        size_t n { matrix.size() }; // define size of matrix
        size_t m { matrix[0].size() }; // define size of matrix
        if (n != m) {
            // logical error
            throw std::logic_error(
                "on-square matrices have no determinant");
        } else {
            // one row matrix
            double determinant {};
            if (n == 1) {
                determinant = matrix[0][0];
            } else if (n == 2) {
                // two row matrix
                determinant = matrix[0][0] * matrix[1][1]
                    - matrix[1][0] * matrix[0][1];
            } else {
                // recursive function
                for (size_t i {}; i < m; i++) {
                    determinant += std::pow(-1, i) * (matrix[0][i])
                        * algebra::determinant(algebra::minor(matrix, 0, i));
                }
            }
            return determinant;
        }
    }
}

// inverse of a Matrix
Matrix algebra::inverse(const Matrix& matrix)
{
    if (matrix.empty()) {
        Matrix Output_Matrix {};
        return Output_Matrix;
    } else {
        size_t n { matrix.size() }; // define size of matrix
        size_t m { matrix[0].size() }; // define size of matrix
        if (n != m) {
            // logical error
            throw std::logic_error(
                "non-square matrices have no inverse");
        } else {
            Matrix Output_Matrix { algebra::zeros(n, m) }; // define matrix
            for (size_t i {}; i < n; i++) {
                for (size_t j {}; j < m; j++) {
                    // determinant of minor
                    Output_Matrix[i][j] = algebra::determinant(
                        algebra::minor(matrix, i, j));
                }
            }
            for (size_t i {}; i < n; i++) {
                for (size_t j {}; j < m; j++) {
                    if (((i + j) % 2) != 0) {
                        // multiply -1
                        Output_Matrix[i][j] = (-1) * Output_Matrix[i][j];
                    }
                }
            }
            // transpose of a matrix
            Output_Matrix = algebra::transpose(Output_Matrix);
            if (algebra::determinant(matrix) == 0) {
                // logical error
                throw std::logic_error(
                    "singular matrices have no inverse");
            } else {
                // devide by determinant
                Output_Matrix = algebra::multiply(Output_Matrix,
                    (1.0 / algebra::determinant(matrix)));
                return Output_Matrix;
            }
        }
    }
}

// concatenate of two Matrices
Matrix algebra::concatenate(const Matrix& matrix1, const Matrix& matrix2,
    int axis = 0)
{
    size_t n1 { matrix1.size() }; // define size of matrix
    size_t m1 { matrix1[0].size() };
    size_t n2 { matrix2.size() };
    size_t m2 { matrix2[0].size() };

    if (axis == 0) {
        if (m1 != m2) {
            // logical error
            throw std::logic_error(
                "matrices with wrong dimensions cannot be concatenated");
        } else {
            // define matrix
            Matrix Output_Matrix { algebra::zeros(n1 + n2, m1) };
            for (size_t i {}; i < n1; i++) {
                for (size_t j {}; j < m1; j++) {
                    Output_Matrix[i][j] = matrix1[i][j];
                }
            }
            for (size_t i {}; i < n2; i++) {
                for (size_t j {}; j < m2; j++) {
                    Output_Matrix[i + n1][j] = matrix2[i][j]; // concat
                }
            }
            return Output_Matrix;
        }

    } else if (axis == 1) {
        if (n1 != n2) {
            // logical error
            throw std::logic_error(
                "matrices with wrong dimensions cannot be concatenated");
        } else {
            // define matrix
            Matrix Output_Matrix { algebra::zeros(n1, m1 + m2) };
            for (size_t i {}; i < n1; i++) {
                for (size_t j {}; j < m1; j++) {
                    Output_Matrix[i][j] = matrix1[i][j];
                }
            }
            for (size_t i {}; i < n2; i++) {
                for (size_t j {}; j < m2; j++) {
                    Output_Matrix[i][j + m1] = matrix2[i][j]; // concat
                }
            }
            return Output_Matrix;
        }
    }
    return Matrix {};
}

// swap two rows
Matrix algebra::ero_swap(const Matrix& matrix, size_t r1, size_t r2)
{
    size_t n { matrix.size() }; // define size of matrix
    size_t m { matrix[0].size() }; // define size of matrix
    Matrix Output_Matrix { algebra::zeros(n, m) }; // define matrix

    if ((r1 >= 0 && r1 < n) && (r2 >= 0 && r2 < n)) {
        // sway the rows
        for (size_t i {}; i < n; i++) {
            if (i == r1) {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[r2][j];
                }
            } else if (i == r2) {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[r1][j];
                }
            } else {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[i][j];
                }
            }
        }
        return Output_Matrix;
    } else {
        // logical error
        throw std::logic_error(
            "r1 or r2 inputs are out of range");
    }
}

// multiply a constant into a rows
Matrix algebra::ero_multiply(const Matrix& matrix, size_t r, double c)
{
    size_t n { matrix.size() }; // define size of matrix
    size_t m { matrix[0].size() }; // define size of matrix
    Matrix Output_Matrix { algebra::zeros(n, m) }; // define matrix

    if ((r >= 0 && r < n)) {
        for (size_t i {}; i < n; i++) {
            if (i == r) {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = c * matrix[i][j]; // multiply
                }
            } else {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[i][j];
                }
            }
        }

        return Output_Matrix;
    } else {
        // logical error
        throw std::logic_error(
            "r inputs are out of range");
    }
}

// multiply a constant into a rows & sum with another row
Matrix algebra::ero_sum(const Matrix& matrix, size_t r1, double c, size_t r2)
{
    size_t n { matrix.size() }; // define size of matrix
    size_t m { matrix[0].size() }; // define size of matrix
    Matrix Output_Matrix { algebra::zeros(n, m) }; // define matrix

    if ((r1 >= 0 && r1 < n) && (r2 >= 0 && r2 < n)) {
        for (size_t i {}; i < n; i++) {
            if (i == r2) {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[i][j] + c * matrix[r1][j]; // multiply
                }
            } else {
                for (size_t j {}; j < m; j++) {
                    Output_Matrix[i][j] = matrix[i][j]; // copy
                }
            }
        }

        return Output_Matrix;
    } else {
        // logical error
        throw std::logic_error(
            "r1 or r2 inputs are out of range");
    }
}

// make an upper triangular Matrix
Matrix algebra::upper_triangular(const Matrix& matrix)
{
    if (matrix.empty()) {
        Matrix Output_Matrix {}; // empty matrix
        return Output_Matrix;
    } else {
        size_t n { matrix.size() }; // define size of matrix
        size_t m { matrix[0].size() }; // define size of matrix
        if (n != m) {
            // logical error
            throw std::logic_error(
                "non-square matrices have no upper triangular form");
        } else {
            Matrix Output_Matrix { matrix };
            for (size_t i {}; i < m; i++) {
                if (Output_Matrix[i][i] == 0.0) {
                    for (size_t j { i + 1 }; j < n; j++) {
                        if (Output_Matrix[j][i] != 0.0) {
                            // swap to make the diag non-zero
                            Output_Matrix = algebra::ero_swap(Output_Matrix, i, j);
                            break;
                        }
                    }
                }
                if (Output_Matrix[i][i] != 0.0) {
                    for (size_t j { i + 1 }; j < n; j++) {
                        // make upper triangular matrix
                        Output_Matrix = algebra::ero_sum(
                            Output_Matrix, i,
                            -(Output_Matrix[j][i]) / (Output_Matrix[i][i]), j);
                    }
                }
            }
            return Output_Matrix;
        }
    }
}
