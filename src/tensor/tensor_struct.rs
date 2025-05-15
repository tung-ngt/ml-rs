use core::f32;
use std::{ops, sync::Arc};

#[derive(Debug, Clone)]
pub struct Tensor<const NO_DIMENSIONS: usize> {
    data: Arc<[f32]>,
    offset: usize,
    strides: [usize; NO_DIMENSIONS],
    shape: [usize; NO_DIMENSIONS],
}

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn get_strides(shape: &[usize; NO_DIMENSIONS]) -> [usize; NO_DIMENSIONS] {
        let mut strides = [1usize; NO_DIMENSIONS];
        for (i, s) in shape.iter().enumerate().skip(1).rev() {
            strides[i - 1] = strides[i] * s;
        }
        strides
    }

    pub fn flat_to_nd_index(
        flat_index: usize,
        shape: &[usize; NO_DIMENSIONS],
    ) -> [usize; NO_DIMENSIONS] {
        let strides = Self::get_strides(shape);
        let mut nd_index = [0; NO_DIMENSIONS];
        let mut remaining = flat_index;
        for (d, dim) in strides.iter().enumerate() {
            nd_index[d] = remaining / dim;
            remaining %= dim;
        }
        assert!(
            remaining == 0,
            "Failed mapping flat_index {} to multidimentional index with strides: {:?}",
            flat_index,
            strides
        );
        nd_index
    }
}

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn new(shape: &[usize; NO_DIMENSIONS]) -> Self {
        let mut size = shape[0];
        for (i, s) in shape.iter().skip(1).enumerate() {
            assert!(*s > 0, "All dimension must > 0. Dimension {} == {}", i, s);
            size *= s;
        }
        assert!(size > 0, "Number of dimensions must be greater than 0");

        let data = vec![0f32; size];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(shape),
            shape: *shape,
        }
    }

    pub fn with_data(
        shape: &[usize; NO_DIMENSIONS],
        strides: &[usize; NO_DIMENSIONS],
        offset: usize,
        data: Arc<[f32]>,
    ) -> Self {
        Self {
            data,
            offset,
            strides: *strides,
            shape: *shape,
        }
    }

    pub fn filled(shape: &[usize; NO_DIMENSIONS], value: f32) -> Self {
        let mut size = shape[0];
        for (i, s) in shape.iter().skip(1).enumerate() {
            assert!(*s > 0, "All dimension must > 0. Dimension {} == {}", i, s);
            size *= s;
        }
        assert!(size > 0, "Number of dimensions must be greater than 0");

        let data = vec![value; size];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(shape),
            shape: *shape,
        }
    }

    pub fn random<G>(shape: &[usize; NO_DIMENSIONS], random_generator: &mut G) -> Self
    where
        G: FnMut() -> f32,
    {
        let mut size = shape[0];
        for (i, s) in shape.iter().skip(1).enumerate() {
            assert!(*s > 0, "All dimension must > 0. Dimension {} == {}", i, s);
            size *= s;
        }
        assert!(size > 0, "Number of dimensions must be greater than 0");

        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(random_generator());
        }
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(shape),
            shape: *shape,
        }
    }

    pub fn empty() -> Self {
        let strides = [1; NO_DIMENSIONS];
        let shape = [1; NO_DIMENSIONS];
        let data = vec![0.0];
        Self {
            data: data.into(),
            offset: 0,
            strides,
            shape,
        }
    }
}

impl Tensor<1> {
    pub fn new_vector(size: usize) -> Self {
        assert!(size > 0, "Size must be greater than 0");

        let data = vec![0f32; size];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: [1],
            shape: [size],
        }
    }

    pub fn vector_with_data(size: usize, stride: usize, offset: usize, data: Arc<[f32]>) -> Self {
        assert!(size > 0, "Size must be greater than 0");
        Self {
            data,
            offset,
            strides: [stride],
            shape: [size],
        }
    }

    pub fn vector_filled(size: usize, value: f32) -> Self {
        assert!(size > 0, "Size must be greater than 0");
        let data = vec![value; size];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: [1],
            shape: [size],
        }
    }

    pub fn vector_random<G>(size: usize, random_generator: &mut G) -> Self
    where
        G: FnMut() -> f32,
    {
        assert!(size > 0, "Size must be greater than 0");
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(random_generator());
        }
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: [1],
            shape: [size],
        }
    }
}

impl Tensor<2> {
    pub fn identity(size: usize) -> Self {
        assert!(size > 0, "Size must be greater than 0");
        let mut data = vec![0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1f32;
        }
        let shape = [size, size];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(&shape),
            shape,
        }
    }

    pub fn new_matrix(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "Rows must be greater than 0");
        assert!(cols > 0, "Cols must be greater than 0");

        let data = vec![0f32; rows * cols];
        let shape = [rows, cols];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(&shape),
            shape,
        }
    }

    pub fn matrix_with_data(
        rows: usize,
        cols: usize,
        strides: &[usize; 2],
        offset: usize,
        data: Arc<[f32]>,
    ) -> Self {
        assert!(rows > 0, "Rows must be greater than 0");
        assert!(cols > 0, "Cols must be greater than 0");

        let shape = [rows, cols];
        Self {
            data,
            offset,
            strides: *strides,
            shape,
        }
    }

    pub fn matrix_filled(rows: usize, cols: usize, value: f32) -> Self {
        assert!(rows > 0, "Rows must be greater than 0");
        assert!(cols > 0, "Cols must be greater than 0");

        let data = vec![value; rows * cols];
        let shape = [rows, cols];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(&shape),
            shape,
        }
    }

    pub fn matrix_random<G>(rows: usize, cols: usize, random_generator: &mut G) -> Self
    where
        G: FnMut() -> f32,
    {
        assert!(rows > 0, "Rows must be greater than 0");
        assert!(cols > 0, "Cols must be greater than 0");

        let size = rows * cols;
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(random_generator());
        }
        let shape = [rows, cols];
        Self {
            data: Arc::from(data),
            offset: 0,
            strides: Self::get_strides(&shape),
            shape,
        }
    }
}

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn no_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize; NO_DIMENSIONS] {
        &self.shape
    }

    pub(super) fn strides(&self) -> &[usize; NO_DIMENSIONS] {
        &self.strides
    }

    pub(super) fn offset(&self) -> usize {
        self.offset
    }

    pub(super) fn data(&self) -> Arc<[f32]> {
        Arc::clone(&self.data)
    }
}

impl<const NO_DIMENSIONS: usize> ops::Index<&[usize; NO_DIMENSIONS]> for Tensor<NO_DIMENSIONS> {
    type Output = f32;

    fn index(&self, index: &[usize; NO_DIMENSIONS]) -> &Self::Output {
        let flat_index = index
            .iter()
            .zip(self.strides)
            .fold(self.offset, |acc, item| acc + item.0 * item.1);
        &self.data[flat_index]
    }
}

impl ops::Index<usize> for Tensor<1> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        let flat_index = self.offset + index * self.strides[0];
        &self.data[flat_index]
    }
}

impl ops::Index<(usize, usize)> for Tensor<2> {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let flat_index = self.offset + index.0 * self.strides[0] + index.1 * self.strides[1];
        &self.data[flat_index]
    }
}

impl ops::Index<(usize, usize, usize)> for Tensor<3> {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let flat_index = self.offset
            + index.0 * self.strides[0]
            + index.1 * self.strides[1]
            + index.2 * self.strides[2];
        &self.data[flat_index]
    }
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn new_tensor() {
        let shape = [1, 2, 3];
        let strides = [6, 3, 1];
        let expected_value = 0f32;

        let a = Tensor::new(&shape);

        assert!(
            a.shape() == &shape,
            "Wrong shape. Expected: {:?} got {:?}",
            &shape,
            a.shape()
        );

        assert!(
            a.strides() == &strides,
            "Wrong strides. Expected: {:?}, got {:?}",
            &strides,
            a.strides()
        );

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let value = a[(i, j, k)];
                    assert!(
                        value == expected_value,
                        "Matrix at {}x{} must be {} got {}",
                        i,
                        j,
                        expected_value,
                        value
                    );
                }
            }
        }
    }

    #[test]
    fn new_filled() {
        let shape = [1, 2, 3];
        let strides = [6, 3, 1];
        let expected_value = 99f32;

        let a = Tensor::filled(&shape, expected_value);

        assert!(
            a.shape() == &shape,
            "Wrong shape. Expected: {:?} got {:?}",
            &shape,
            a.shape()
        );

        assert!(
            a.strides() == &strides,
            "Wrong strides true: {:?}, got {:?}",
            &strides,
            a.strides()
        );

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let value = a[(i, j, k)];
                    assert!(
                        value == expected_value,
                        "Matrix at {}x{} must be {} got {}",
                        i,
                        j,
                        expected_value,
                        value
                    );
                }
            }
        }
    }

    #[test]
    fn new_identity() {
        let size = 3;
        let strides = [3, 1];
        let shape = [3, 3];

        let a = Tensor::identity(size);

        assert!(
            a.shape() == &shape,
            "Wrong shape. Expected: {:?} got {:?}",
            &shape,
            a.shape()
        );
        assert!(
            a.strides == strides,
            "Wrong strides true: {:?}, got {:?}",
            strides,
            a.strides
        );

        for i in 0..size {
            for j in 0..size {
                let value = a[(i, j)];
                let expected_value = if i == j { 1f32 } else { 0f32 };
                assert!(
                    value == expected_value,
                    "Identity matrix at {}x{} must be {} got {}",
                    i,
                    j,
                    expected_value,
                    value
                );
            }
        }
    }

    #[test]
    fn new_with_data() {
        let shape = [2, 2, 2];
        let strides = [4, 2, 1];
        let data = (0..8).map(|x| x as f32).collect();

        let a = Tensor::with_data(&shape, &strides, 0, data);

        assert!(
            a.shape() == &shape,
            "Wrong shape. Expected: {:?} got {:?}",
            &shape,
            a.shape()
        );
        let mut expected_value = 0f32;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let value = a[(i, j, k)];
                    assert!(
                        value == expected_value,
                        "Matrix at {}x{} must be {} got {}",
                        i,
                        j,
                        expected_value,
                        value
                    );
                    expected_value += 1f32;
                }
            }
        }
    }

    #[test]
    fn new_vec() {
        let size = 3;
        let strides = [1];
        let expected_value = 99f32;

        let a = Tensor::vector_filled(3, expected_value);

        assert!(
            a.shape() == &[size],
            "Wrong shape. Expected: {:?} got {:?}",
            &[size],
            a.shape()
        );

        assert!(
            a.strides() == &strides,
            "Wrong strides true: {:?}, got {:?}",
            &strides,
            a.strides()
        );

        for i in 0..size {
            let value = a[i];
            assert!(
                value == expected_value,
                "Vector at {} must be {} got {}",
                i,
                expected_value,
                value
            );
        }
    }
}
