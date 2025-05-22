use super::Tensor;
use std::{array, ops, sync::Arc};

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn add(&self, b_tensor: &Self) -> Self {
        let a_tensor = self;
        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();

        assert!(
            a_shape == b_shape,
            "Cannot add tensor {a_shape:?} with tensor {b_shape:?}"
        );

        let strides = Self::get_strides(a_shape);
        let no_elements = a_shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}. Remaining: {}. index {:?}",
                flat_index,
                a_shape,
                remaining,
                multi_dim_index
            );
            new_data[flat_index] = a_tensor[&multi_dim_index] + b_tensor[&multi_dim_index]
        }

        Self::with_data(a_shape, &strides, 0, Arc::from(new_data))
    }

    pub fn sub(&self, b_tensor: &Self) -> Self {
        let a_tensor = self;
        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();

        assert!(
            a_shape == b_shape,
            "Cannot sub tensor {a_shape:?} with tensor {b_shape:?}"
        );

        let strides = Self::get_strides(a_shape);
        let no_elements = a_shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                a_shape
            );

            new_data[flat_index] = a_tensor[&multi_dim_index] - b_tensor[&multi_dim_index]
        }

        Self::with_data(a_shape, &strides, 0, Arc::from(new_data))
    }

    pub fn mul_elem(&self, b_tensor: &Self) -> Self {
        let a_tensor = self;
        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();

        assert!(
            a_shape == b_shape,
            "Cannot mul tensor {a_shape:?} with tensor {b_shape:?}"
        );

        let strides = Self::get_strides(a_shape);
        let no_elements = a_shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                a_shape
            );

            new_data[flat_index] = a_tensor[&multi_dim_index] * b_tensor[&multi_dim_index]
        }

        Self::with_data(a_shape, &strides, 0, Arc::from(new_data))
    }

    pub fn div_elem(&self, b_tensor: &Self) -> Self {
        let a_tensor = self;
        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();

        assert!(
            a_shape == b_shape,
            "Cannot div tensor {a_shape:?} with tensor {b_shape:?}"
        );

        let strides = Self::get_strides(a_shape);
        let no_elements = a_shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                a_shape
            );

            new_data[flat_index] = a_tensor[&multi_dim_index] / b_tensor[&multi_dim_index]
        }

        Self::with_data(a_shape, &strides, 0, Arc::from(new_data))
    }

    pub fn equals(&self, b_tensor: &Self) -> bool {
        let a_tensor = self;
        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();

        if a_shape != b_shape {
            return false;
        };

        let strides = Self::get_strides(a_shape);
        let no_elements = a_shape.iter().product();
        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                a_shape
            );

            if !a_tensor[&multi_dim_index].is_finite() || !b_tensor[&multi_dim_index].is_finite() {
                return false;
            }
            if (a_tensor[&multi_dim_index] - b_tensor[&multi_dim_index]).abs() >= 1e-6 {
                return false;
            }
        }
        true
    }

    pub fn scale(&self, scaler: f32) -> Self {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = self[&multi_dim_index] * scaler;
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn add_value(&self, value: f32) -> Self {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = self[&multi_dim_index] + value;
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn value_sub(value: f32, tensor: &Self) -> Self {
        let shape = tensor.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = value - tensor[&multi_dim_index];
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn value_div(value: f32, tensor: &Self) -> Self {
        let shape = tensor.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = value / tensor[&multi_dim_index];
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn powf(&self, value: f32) -> Self {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = self[&multi_dim_index].powf(value);
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn powi(&self, value: i32) -> Self {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = self[&multi_dim_index].powi(value);
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn apply<T>(&self, map: T) -> Self
    where
        T: Fn(f32) -> f32,
    {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = map(self[&multi_dim_index]);
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn apply_mut<T>(&self, mut map: T) -> Self
    where
        T: FnMut(f32) -> f32,
    {
        let shape = self.shape();
        let strides = Self::get_strides(shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in strides.iter().enumerate() {
                multi_dim_index[d] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}",
                flat_index,
                shape
            );

            new_data[flat_index] = map(self[&multi_dim_index]);
        }

        Self::with_data(shape, &strides, 0, Arc::from(new_data))
    }

    pub fn transpose(&self, new_dims: &[usize; NO_DIMENSIONS]) -> Self {
        let shape = self.shape();
        let new_shape: [usize; NO_DIMENSIONS] = array::from_fn(|i| shape[new_dims[i]]);

        let new_strides = Self::get_strides(&new_shape);
        let no_elements = shape.iter().product();

        let mut new_data = vec![0f32; no_elements];

        let mut old_multi_dim_index = [0; NO_DIMENSIONS];
        let mut remaining: usize;
        for flat_index in 0..no_elements {
            remaining = flat_index;
            for (d, dim) in new_strides.iter().enumerate() {
                old_multi_dim_index[new_dims[d]] = remaining / dim;
                remaining %= dim;
            }
            assert!(
                remaining == 0,
                "Failed mapping flat_index {} to multidimentional index {:?}. Remaining: {}. index {:?}",
                flat_index,
                new_shape,
                remaining,
                old_multi_dim_index
            );

            new_data[flat_index] = self[&old_multi_dim_index];
        }

        Self::with_data(&new_shape, &new_strides, 0, Arc::from(new_data))
    }

    pub fn sum(&self) -> f32 {
        let mut s = 0f32;
        let mut multi_dim_index: [usize; NO_DIMENSIONS];
        let shape = self.shape();
        for flat_index in 0..self.no_elements() {
            multi_dim_index = Self::flat_to_nd_index(flat_index, shape);
            s += self[&multi_dim_index]
        }
        s
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.no_elements() as f32
    }

    pub fn std(&self) -> f32 {
        let no_elements = self.no_elements() as f32;
        let sum = self.sum();
        let mean = sum / no_elements;

        let diff = self - mean;
        let square_diff = diff.powi(2);
        let sum_square_diff = square_diff.sum();
        let mean_square_diff = sum_square_diff / (no_elements + 1.0);

        mean_square_diff.sqrt()
    }

    pub fn sum_dim(&self, dim: usize) -> Self {
        let shape = self.shape();
        let mut new_shape = *shape;
        new_shape[dim] = 1;
        let strides = Self::get_strides(&new_shape);

        let mut data = vec![0.0; new_shape.iter().product()];
        for flat_index in 0..self.no_elements() {
            let multi_dim_index = Self::flat_to_nd_index(flat_index, shape);
            let mut sum_index = multi_dim_index;
            sum_index[dim] = 1;
            let flat_index = sum_index.iter().zip(strides).fold(0, |a, (i, s)| a + i * s);
            data[flat_index] += self[&multi_dim_index]
        }
        Self::with_data(&new_shape, &strides, 0, data.into())
    }
}

impl Tensor<2> {
    pub fn row_add(&self, vector: &Tensor<1>) -> Tensor<2> {
        let matrix = self;
        let &[rows, cols] = matrix.shape();
        let &[vector_size] = vector.shape();

        assert!(
            cols == vector_size,
            "Cannot add row matrix {}x{} with vector ({},)",
            rows,
            cols,
            vector_size
        );

        let stride = cols;
        let mut new_data = vec![0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = matrix[(i, j)] + vector[j];
            }
        }

        Tensor::matrix_with_data(rows, cols, &[stride, 1], 0, Arc::from(new_data))
    }

    pub fn col_add(&self, vector: &Tensor<1>) -> Tensor<2> {
        let matrix = self;
        let &[rows, cols] = matrix.shape();
        let &[vector_size] = vector.shape();

        assert!(
            rows == vector_size,
            "Cannot add col matrix {}x{} with vector ({},)",
            rows,
            cols,
            vector_size
        );

        let stride = cols;
        let mut new_data = vec![0f32; rows * cols];

        for j in 0..cols {
            for i in 0..rows {
                new_data[i * stride + j] = matrix[(i, j)] + vector[i];
            }
        }

        Tensor::matrix_with_data(rows, cols, &[stride, 1], 0, Arc::from(new_data))
    }

    pub fn row_mul(&self, vector: &Tensor<1>) -> Tensor<2> {
        let matrix = self;
        let &[rows, cols] = matrix.shape();
        let &[vector_size] = vector.shape();

        assert!(
            cols == vector_size,
            "Cannot mul row matrix {}x{} with vector ({},)",
            rows,
            cols,
            vector_size
        );

        let stride = cols;
        let mut new_data = vec![0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                new_data[i * stride + j] = matrix[(i, j)] * vector[j];
            }
        }

        Tensor::matrix_with_data(rows, cols, &[stride, 1], 0, Arc::from(new_data))
    }

    pub fn col_mul(&self, vector: &Tensor<1>) -> Tensor<2> {
        let matrix = self;
        let &[rows, cols] = matrix.shape();
        let &[vector_size] = vector.shape();

        assert!(
            rows == vector_size,
            "Cannot mul col matrix {}x{} with vector ({},)",
            rows,
            cols,
            vector_size
        );

        let stride = cols;
        let mut new_data = vec![0f32; rows * cols];

        for j in 0..cols {
            for i in 0..rows {
                new_data[i * stride + j] = matrix[(i, j)] * vector[i];
            }
        }

        Tensor::matrix_with_data(rows, cols, &[stride, 1], 0, Arc::from(new_data))
    }
}

impl Tensor<1> {
    pub fn dot(&self, b_vector: &Tensor<1>) -> Tensor<1> {
        let a_vector = self;
        let &[a_size] = self.shape();
        let &[b_size] = b_vector.shape();

        assert!(
            a_size == b_size,
            "Cannot dot vector ({},) with vector ({},)",
            a_size,
            b_size
        );

        let a_data = a_vector.data();
        let b_data = b_vector.data();

        let dot_value = a_data
            .iter()
            .zip(b_data.iter())
            .fold(0f32, |acc, (a, b)| acc + (a * b));

        Tensor::vector_filled(1, dot_value)
    }

    pub fn to_diag(&self) -> Tensor<2> {
        let no_elements = self.no_elements();
        let mut data = vec![0.0; no_elements * no_elements];

        for i in 0..no_elements {
            data[i * no_elements + i] = self[i];
        }

        Tensor::matrix_with_data(no_elements, no_elements, &[no_elements, 1], 0, data.into())
    }
}

impl Tensor<2> {
    pub fn matmul(&self, b_matrix: &Tensor<2>) -> Tensor<2> {
        let a_matrix = self;
        let &[a_rows, a_cols] = a_matrix.shape();
        let &[b_rows, b_cols] = b_matrix.shape();

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
        Tensor::<2>::matrix_with_data(a_rows, b_cols, &[stride, 1], 0, Arc::from(data))
    }

    pub fn t(&self) -> Tensor<2> {
        let &[rows, cols] = self.shape();

        let mut new_data = vec![0f32; rows * cols];

        let stride = rows;
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * stride + i] = self[(i, j)];
            }
        }
        Tensor::<2>::matrix_with_data(cols, rows, &[stride, 1], 0, Arc::from(new_data))
    }

    pub fn sum_row(&self) -> Tensor<1> {
        let &[rows, cols] = self.shape();

        let mut new_data = vec![0f32; cols];
        for i in 0..rows {
            for j in 0..cols {
                new_data[j] += self[(i, j)];
            }
        }

        Tensor::vector_with_data(cols, 1, 0, Arc::from(new_data))
    }

    pub fn sum_col(&self) -> Tensor<1> {
        let &[rows, cols] = self.shape();

        let mut new_data = vec![0f32; rows];
        for i in 0..rows {
            for j in 0..cols {
                new_data[i] += self[(i, j)];
            }
        }

        Tensor::vector_with_data(rows, 1, 0, Arc::from(new_data))
    }

    pub fn max_col(&self) -> (Tensor<1>, Tensor<1>) {
        let &[rows, cols] = self.shape();

        let mut max_data = vec![f32::NEG_INFINITY; rows];
        let mut index_data = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                let value = self[(i, j)];
                if value >= max_data[i] {
                    max_data[i] = value;
                    index_data[i] = j as f32;
                }
            }
        }

        (
            Tensor::vector_with_data(rows, 1, 0, max_data.into()),
            Tensor::vector_with_data(rows, 1, 0, index_data.into()),
        )
    }

    pub fn max_row(&self) -> (Tensor<1>, Tensor<1>) {
        let &[rows, cols] = self.shape();

        let mut max_data = vec![f32::NEG_INFINITY; cols];
        let mut index_data = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                let value = self[(i, j)];
                if value >= max_data[j] {
                    max_data[j] = value;
                    index_data[j] = i as f32;
                }
            }
        }

        (
            Tensor::vector_with_data(cols, 1, 0, max_data.into()),
            Tensor::vector_with_data(cols, 1, 0, index_data.into()),
        )
    }

    pub fn min_col(&self) -> (Tensor<1>, Tensor<1>) {
        let &[rows, cols] = self.shape();

        let mut max_data = vec![f32::INFINITY; rows];
        let mut index_data = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                let value = self[(i, j)];
                if value <= max_data[i] {
                    max_data[i] = value;
                    index_data[i] = j as f32;
                }
            }
        }

        (
            Tensor::vector_with_data(rows, 1, 0, max_data.into()),
            Tensor::vector_with_data(rows, 1, 0, index_data.into()),
        )
    }

    pub fn min_row(&self) -> (Tensor<1>, Tensor<1>) {
        let &[rows, cols] = self.shape();

        let mut max_data = vec![f32::INFINITY; cols];
        let mut index_data = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                let value = self[(i, j)];
                if value <= max_data[j] {
                    max_data[j] = value;
                    index_data[j] = i as f32;
                }
            }
        }

        (
            Tensor::vector_with_data(cols, 1, 0, max_data.into()),
            Tensor::vector_with_data(cols, 1, 0, index_data.into()),
        )
    }
}

impl Tensor<3> {
    pub fn batch_matmul(&self, b_matrix: &Tensor<2>) -> Tensor<3> {
        let a_matrix = self;
        let &[batch, a_rows, a_cols] = a_matrix.shape();
        let &[b_rows, b_cols] = b_matrix.shape();

        assert!(
            a_cols == b_rows,
            "Cannot dot a_cols {} must equals b_rows {}",
            a_cols,
            b_rows
        );

        let output_shape = [batch, a_rows, b_cols];
        let strides = Tensor::get_strides(&output_shape);
        let inner_dimension = a_cols;

        let mut data = vec![0f32; output_shape.iter().product()];

        for b in 0..batch {
            for i in 0..a_rows {
                for j in 0..b_cols {
                    for k in 0..inner_dimension {
                        data[b * strides[0] + i * strides[1] + j] +=
                            a_matrix[(b, i, k)] * b_matrix[(k, j)]
                    }
                }
            }
        }
        Tensor::with_data(&output_shape, &strides, 0, data.into())
    }
}

impl ops::Mul for &Tensor<2> {
    type Output = Tensor<2>;
    fn mul(self, b_tensor: Self) -> Self::Output {
        Tensor::<2>::matmul(self, b_tensor)
    }
}

impl ops::Mul for &Tensor<1> {
    type Output = Tensor<1>;
    fn mul(self, b_vector: Self) -> Self::Output {
        Tensor::<1>::dot(self, b_vector)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Add for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn add(self, b_tensor: Self) -> Self::Output {
        Tensor::add(self, b_tensor)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Sub for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn sub(self, b_tensor: Self) -> Self::Output {
        Tensor::sub(self, b_tensor)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Mul<f32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn mul(self, scaler: f32) -> Self::Output {
        Tensor::scale(self, scaler)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Mul<&Tensor<NO_DIMENSIONS>> for f32 {
    type Output = Tensor<NO_DIMENSIONS>;
    fn mul(self, tensor: &Tensor<NO_DIMENSIONS>) -> Self::Output {
        Tensor::scale(tensor, self)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Div<f32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn div(self, scaler: f32) -> Self::Output {
        Tensor::scale(self, 1f32 / scaler)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Div<&Tensor<NO_DIMENSIONS>> for f32 {
    type Output = Tensor<NO_DIMENSIONS>;
    fn div(self, tensor: &Tensor<NO_DIMENSIONS>) -> Self::Output {
        Tensor::value_div(self, tensor)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Add<f32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn add(self, value: f32) -> Self::Output {
        Tensor::add_value(self, value)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Add<&Tensor<NO_DIMENSIONS>> for f32 {
    type Output = Tensor<NO_DIMENSIONS>;
    fn add(self, tensor: &Tensor<NO_DIMENSIONS>) -> Self::Output {
        Tensor::add_value(tensor, self)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Sub<f32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn sub(self, value: f32) -> Self::Output {
        Tensor::add_value(self, -value)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Sub<&Tensor<NO_DIMENSIONS>> for f32 {
    type Output = Tensor<NO_DIMENSIONS>;
    fn sub(self, tensor: &Tensor<NO_DIMENSIONS>) -> Self::Output {
        Tensor::value_sub(self, tensor)
    }
}

impl<const NO_DIMENSIONS: usize> ops::BitXor<f32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn bitxor(self, value: f32) -> Self::Output {
        Tensor::powf(self, value)
    }
}

impl<const NO_DIMENSIONS: usize> ops::BitXor<i32> for &Tensor<NO_DIMENSIONS> {
    type Output = Tensor<NO_DIMENSIONS>;
    fn bitxor(self, value: i32) -> Self::Output {
        Tensor::powi(self, value)
    }
}

impl<const NO_DIMENSIONS: usize> PartialEq for Tensor<NO_DIMENSIONS> {
    fn eq(&self, other: &Self) -> bool {
        Tensor::equals(self, other)
    }
}

#[cfg(test)]
mod tensor_ops_tests {
    use core::f32;
    use std::sync::Arc;

    use crate::tensor::Tensor;

    #[test]
    fn test_eq() {
        let shape = [4, 5, 6];
        let a = Tensor::new(&shape);
        let b = Tensor::filled(&shape, 123f32);
        let c = Tensor::filled(&shape, -3f32);
        let d = Tensor::filled(&shape, f32::NAN);
        assert!(a != b);
        assert!(a != c);
        assert!(b != c);
        assert!(a == a);
        assert!(b == b);
        assert!(c == c);
        assert!(a != d);
        assert!(d != d);
    }

    #[test]
    fn add_tensors() {
        let shape = [5, 5, 5];
        let a = Tensor::filled(&shape, 1f32);
        let b = Tensor::filled(&shape, 9f32);
        let expected = Tensor::filled(&shape, 10f32);
        let c = &a + &b;
        assert!(expected == c);
    }

    #[test]
    fn sub_tensors() {
        let shape = [5, 5, 5];
        let a = Tensor::filled(&shape, 1f32);
        let b = Tensor::filled(&shape, 9f32);
        let expected = Tensor::filled(&shape, -8f32);
        let c = &a - &b;
        assert!(expected == c);
    }

    #[test]
    fn mul_elem_tensors() {
        let shape = [5, 5, 5];
        let a = Tensor::filled(&shape, -2f32);
        let b = Tensor::filled(&shape, 3f32);
        let expected = Tensor::filled(&shape, -6f32);
        let c = a.mul_elem(&b);
        assert!(expected == c);
    }

    #[test]
    fn div_elem_tensors() {
        let shape = [5, 5, 5];
        let a = Tensor::filled(&shape, -9f32);
        let b = Tensor::filled(&shape, 3f32);
        let expected = Tensor::filled(&shape, -3f32);
        let c = a.div_elem(&b);
        assert!(expected == c);
    }

    #[test]
    fn scale_tensors() {
        let shape = [3, 10, 2];
        let a = Tensor::filled(&shape, 1f32);
        let expected = Tensor::filled(&shape, 6f32);
        let c = &a * 6f32;
        assert!(expected == c);
    }

    #[test]
    fn add_value_tensors() {
        let shape = [3, 10, 2];
        let a = Tensor::filled(&shape, 1f32);
        let expected = Tensor::filled(&shape, 3f32);
        let c = &a + 2f32;
        assert!(expected == c);
    }

    #[test]
    fn value_sub_tensors() {
        let shape = [3, 10, 2];
        let a = Tensor::filled(&shape, 1f32);
        let expected = Tensor::filled(&shape, -2f32);
        let c = &a - 3f32;
        assert!(expected == c);
    }

    #[test]
    fn powi_tensors() {
        let shape = [3, 10, 2];
        let a = Tensor::filled(&shape, 2f32);
        let expected = Tensor::filled(&shape, 8f32);
        let c = &a ^ 3;
        assert!(expected == c);
    }

    #[test]
    fn powf_tensors() {
        let shape = [3, 10, 2];
        let a = Tensor::filled(&shape, 4f32);
        let expected = Tensor::filled(&shape, 2f32);
        let c = &a ^ 0.5;
        assert!(expected == c);
    }

    #[test]
    fn apply_tensors() {
        let a = Tensor::new_vector(10);
        let b = a.apply(|_| 20f32);

        for i in 0..10 {
            assert!(b[i] == 20f32);
        }
    }

    #[rustfmt::skip]
    #[test]
    fn transpose_tensors() {
        let shape_012 = [2, 3, 4];
        let shape_021 = [2, 4, 3];
        let shape_201 = [4, 2, 3];
        let shape_210 = [4, 3, 2];
        let shape_120 = [3, 4, 2];
        let shape_102 = [3, 2, 4];

        let data_012 = vec![
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,

            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0
        ];
        let data_021 = vec![
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 
            4.0, 4.0, 4.0,

            -1.0, -1.0, -1.0,
            -2.0, -2.0, -2.0,
            -3.0, -3.0, -3.0, 
            -4.0, -4.0, -4.0,
        ];
        let data_201 = vec![
            1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0,

            2.0, 2.0, 2.0,
            -2.0, -2.0, -2.0,

            3.0, 3.0, 3.0, 
            -3.0, -3.0, -3.0, 

            4.0, 4.0, 4.0,
            -4.0, -4.0, -4.0,
        ];
        let data_210 = vec![
            1.0, -1.0,
            1.0, -1.0,
            1.0, -1.0,

            2.0, -2.0,
            2.0, -2.0,
            2.0, -2.0,

            3.0, -3.0,
            3.0, -3.0,
            3.0, -3.0,

            4.0, -4.0,
            4.0, -4.0,
            4.0, -4.0,
        ];
        let data_120 = vec![
            1.0, -1.0, 
            2.0, -2.0, 
            3.0, -3.0, 
            4.0, -4.0, 

            1.0, -1.0, 
            2.0, -2.0, 
            3.0, -3.0, 
            4.0, -4.0, 

            1.0, -1.0, 
            2.0, -2.0, 
            3.0, -3.0, 
            4.0, -4.0, 
        ];
        let data_102 = vec![
            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,

            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,

            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
    
        let a_012 = Tensor::with_data(&shape_012, &Tensor::get_strides(&shape_012), 0, Arc::from(data_012));
        let a_021 = Tensor::with_data(&shape_021, &Tensor::get_strides(&shape_021), 0, Arc::from(data_021));
        let a_201 = Tensor::with_data(&shape_201, &Tensor::get_strides(&shape_201), 0, Arc::from(data_201));
        let a_210 = Tensor::with_data(&shape_210, &Tensor::get_strides(&shape_210), 0, Arc::from(data_210));
        let a_120 = Tensor::with_data(&shape_120, &Tensor::get_strides(&shape_120), 0, Arc::from(data_120));
        let a_102 = Tensor::with_data(&shape_102, &Tensor::get_strides(&shape_102), 0, Arc::from(data_102));

        assert!(a_012.transpose(&[0, 2, 1]) == a_021);
        assert!(a_012.transpose(&[2, 0, 1]) == a_201);
        assert!(a_012.transpose(&[2, 1, 0]) == a_210);
        assert!(a_012.transpose(&[1, 2, 0]) == a_120);
        assert!(a_012.transpose(&[1, 0, 2]) == a_102);

        assert!(a_012.transpose(&[0, 2, 1]).shape() == &shape_021);
        assert!(a_012.transpose(&[2, 0, 1]).shape() == &shape_201);
        assert!(a_012.transpose(&[2, 1, 0]).shape() == &shape_210);
        assert!(a_012.transpose(&[1, 2, 0]).shape() == &shape_120);
        assert!(a_012.transpose(&[1, 0, 2]).shape() == &shape_102);
    }

    #[test]
    fn apply_mut_tensors() {
        let a = Tensor::vector_filled(10, 1f32);
        let mut count = -1f32;
        let b = a.apply_mut(|x| {
            count += 1f32;
            x + count
        });

        for i in 0..10 {
            assert!(b[i] == (i as f32 + 1f32));
        }
    }

    #[test]
    fn dot_matrix() {
        let i = Tensor::identity(3);
        let a = Tensor::matrix_filled(3, 3, 3f32);
        let b = &i * &a;
        let c = &a * &i;
        let d = Tensor::matrix_filled(3, 2, 3f32);
        let e = &a * &d;
        let f = Tensor::matrix_filled(3, 2, 27f32);
        assert!(a == b);
        assert!(a == c);
        assert!(b == c);
        assert!(e == f);
    }

    #[rustfmt::skip]
    #[test]
    fn transpose_matrix() {
        let rows = 3;
        let cols = 2;
        let data = vec![
            1.0, 2.0,
            3.0, 3.0,
            5.0, 6.0
        ];
        let tranposed_data = vec![
            1.0, 3.0, 5.0,
            2.0, 3.0, 6.0
        ];

        let a = Tensor::matrix_with_data(rows, cols, &[2, 1], 0, Arc::from(data));
        let b = Tensor::matrix_with_data(cols, rows, &[3, 1], 0, Arc::from(tranposed_data));
        assert!(a.t() == b);
        assert!(a == b.t());
    }
}
