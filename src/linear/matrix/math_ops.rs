use super::Matrix;
use std::{ops, sync::Arc};

impl Matrix {
    pub fn dot(&self, b_matrix: &Matrix) -> Matrix {
        let a_matrix = self;
        let (a_rows, a_cols) = a_matrix.shape();
        let (b_rows, b_cols) = b_matrix.shape();

        assert!(
            a_cols == b_rows,
            "Cannot dot a_cols {} must equals b_rows {}",
            a_cols,
            b_rows
        );

        let inner_dimension = a_cols;
        let stride = b_cols;

        let mut data = vec![0f32; a_rows * b_cols];

        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..inner_dimension {
                    data[i * stride + j] += a_matrix[(i, k)] * b_matrix[(k, j)]
                }
            }
        }
        Matrix::with_data(a_rows, b_cols, stride, 0, Arc::from(data))
    }

    pub fn add(&self, b_matrix: &Matrix) -> Matrix {
        let a_matrix = self;
        let (a_rows, a_cols) = a_matrix.shape();
        let (b_rows, b_cols) = b_matrix.shape();

        assert!(
            (a_rows == b_rows) && (a_cols == b_cols),
            "Cannot add matrix {a_rows}x{a_cols} with matrix {b_rows}x{b_cols}",
        );

        let stride = a_cols;

        let mut data = vec![0f32; a_rows * a_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                data[i * stride + j] = a_matrix[(i, j)] + b_matrix[(i, j)]
            }
        }
        Matrix::with_data(a_rows, a_cols, stride, 0, Arc::from(data))
    }

    pub fn sub(&self, b_matrix: &Matrix) -> Matrix {
        let a_matrix = self;
        let (a_rows, a_cols) = a_matrix.shape();
        let (b_rows, b_cols) = b_matrix.shape();

        assert!(
            (a_rows == b_rows) && (a_cols == b_cols),
            "Cannot add matrix {a_rows}x{a_cols} with matrix {b_rows}x{b_cols}",
        );

        let stride = a_cols;
        let mut data = vec![0f32; a_rows * a_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                data[i * stride + j] = a_matrix[(i, j)] - b_matrix[(i, j)]
            }
        }
        Matrix::with_data(a_rows, a_cols, stride, 0, Arc::from(data))
    }

    pub fn scale(&self, scaler: f32) -> Matrix {
        let (rows, cols) = self.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = self[(i, j)] * scaler;
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }

    pub fn add_value(&self, value: f32) -> Matrix {
        let (rows, cols) = self.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = self[(i, j)] + value;
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }

    pub fn value_sub(value: f32, matrix: &Matrix) -> Matrix {
        let (rows, cols) = matrix.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = value - matrix[(i, j)];
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }

    pub fn t(&self) -> Matrix {
        let (rows, cols) = self.shape();

        let mut new_data = vec![0f32; rows * cols];

        let stride = rows;
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * stride + i] = self[(i, j)];
            }
        }
        Matrix::with_data(cols, rows, stride, 0, Arc::from(new_data))
    }

    pub fn mul_elem(&self, b_matrix: &Matrix) -> Matrix {
        let a_matrix = self;
        let (a_rows, a_cols) = a_matrix.shape();
        let (b_rows, b_cols) = b_matrix.shape();

        assert!(
            (a_rows == b_rows) && (a_cols == b_cols),
            "Cannot mul element wise matrix {a_rows}x{a_cols} with matrix {b_rows}x{b_cols}",
        );

        let stride = a_cols;
        let mut new_data = vec![0f32; a_rows * a_cols];
        for i in 0..a_rows {
            for j in 0..a_cols {
                new_data[i * stride + j] = a_matrix[(i, j)] * b_matrix[(i, j)];
            }
        }

        Matrix::with_data(a_rows, a_cols, stride, 0, Arc::from(new_data))
    }

    pub fn div_elem(&self, b_matrix: &Matrix) -> Matrix {
        let a_matrix = self;
        let (a_rows, a_cols) = a_matrix.shape();
        let (b_rows, b_cols) = b_matrix.shape();

        assert!(
            (a_rows == b_rows) && (a_cols == b_cols),
            "Cannot mul element wise matrix {a_rows}x{a_cols} with matrix {b_rows}x{b_cols}",
        );

        let stride = a_cols;
        let mut new_data = vec![0f32; a_rows * a_cols];
        for i in 0..a_rows {
            for j in 0..a_cols {
                new_data[i * stride + j] = a_matrix[(i, j)] / b_matrix[(i, j)];
            }
        }

        Matrix::with_data(a_rows, a_cols, stride, 0, Arc::from(new_data))
    }

    pub fn powf(&self, value: f32) -> Matrix {
        let (rows, cols) = self.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = self[(i, j)].powf(value);
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }

    pub fn powi(&self, value: i32) -> Matrix {
        let (rows, cols) = self.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = self[(i, j)].powi(value);
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }

    pub fn apply<T>(&self, map: T) -> Matrix
    where
        T: Fn(f32) -> f32,
    {
        let (rows, cols) = self.shape();
        let mut new_data = vec![0f32; rows * cols];

        let stride = cols;
        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = map(self[(i, j)]);
            }
        }
        Matrix::with_data(rows, cols, stride, 0, Arc::from(new_data))
    }
}

impl ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, b_matrix: Self) -> Self::Output {
        Matrix::dot(self, b_matrix)
    }
}

impl ops::Add for &Matrix {
    type Output = Matrix;

    fn add(self, b_matrix: Self) -> Self::Output {
        Matrix::add(self, b_matrix)
    }
}

impl ops::Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, b_matrix: Self) -> Self::Output {
        Matrix::sub(self, b_matrix)
    }
}

impl ops::Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, scaler: f32) -> Self::Output {
        Matrix::scale(self, scaler)
    }
}

impl ops::Mul<&Matrix> for f32 {
    type Output = Matrix;
    fn mul(self, matrix: &Matrix) -> Self::Output {
        Matrix::scale(matrix, self)
    }
}

impl ops::Div<f32> for &Matrix {
    type Output = Matrix;
    fn div(self, scaler: f32) -> Self::Output {
        Matrix::scale(self, 1f32 / scaler)
    }
}

impl ops::Div<&Matrix> for f32 {
    type Output = Matrix;
    fn div(self, matrix: &Matrix) -> Self::Output {
        Matrix::scale(matrix, 1f32 / self)
    }
}

impl ops::Add<f32> for &Matrix {
    type Output = Matrix;
    fn add(self, value: f32) -> Self::Output {
        Matrix::add_value(self, value)
    }
}

impl ops::Add<&Matrix> for f32 {
    type Output = Matrix;
    fn add(self, matrix: &Matrix) -> Self::Output {
        Matrix::add_value(matrix, self)
    }
}

impl ops::Sub<f32> for &Matrix {
    type Output = Matrix;
    fn sub(self, value: f32) -> Self::Output {
        Matrix::add_value(self, -value)
    }
}

impl ops::Sub<&Matrix> for f32 {
    type Output = Matrix;
    fn sub(self, matrix: &Matrix) -> Self::Output {
        Matrix::value_sub(self, matrix)
    }
}

impl ops::BitXor<f32> for &Matrix {
    type Output = Matrix;
    fn bitxor(self, value: f32) -> Self::Output {
        Matrix::powf(self, value)
    }
}

impl ops::BitXor<i32> for &Matrix {
    type Output = Matrix;
    fn bitxor(self, value: i32) -> Self::Output {
        Matrix::powi(self, value)
    }
}
