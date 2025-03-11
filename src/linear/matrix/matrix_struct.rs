use std::{ops, sync::Arc};

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    stride: usize,
    offset: usize,
    data: Arc<[f32]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0f32; rows * cols];
        Self {
            rows,
            cols,
            stride: cols,
            offset: 0,
            data: Arc::from(data),
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut data = vec![0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1f32;
        }
        Self {
            rows: size,
            cols: size,
            stride: size,
            offset: 0,
            data: Arc::from(data),
        }
    }

    pub fn with_data(
        rows: usize,
        cols: usize,
        stride: usize,
        offset: usize,
        data: Arc<[f32]>,
    ) -> Self {
        Self {
            rows,
            cols,
            stride,
            offset,
            data,
        }
    }

    pub fn filled(rows: usize, cols: usize, value: f32) -> Self {
        let data = vec![value; rows * cols];
        Self {
            rows,
            cols,
            stride: cols,
            offset: 0,
            data: Arc::from(data),
        }
    }
}

impl Matrix {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    pub(super) fn offset(&self) -> usize {
        self.offset
    }

    pub(super) fn data(&self) -> Arc<[f32]> {
        Arc::clone(&self.data)
    }
}

impl ops::Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.offset() + index.0 * self.stride() + index.1]
    }
}
