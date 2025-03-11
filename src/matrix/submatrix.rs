use super::Matrix;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

impl Matrix {
    pub fn row(&self, index: usize) -> Matrix {
        let (rows, cols) = self.shape();
        assert!(
            index < rows,
            "Row index {} out of range for matrix {}x{}",
            index,
            rows,
            cols
        );

        let stride = self.stride();
        Matrix::with_data(1, cols, stride, self.offset() + index * stride, self.data())
    }

    pub fn col(&self, index: usize) -> Matrix {
        let (rows, cols) = self.shape();
        assert!(
            index < cols,
            "Col index {} out of range for matrix {}x{}",
            index,
            rows,
            cols
        );

        Matrix::with_data(rows, 1, self.stride(), self.offset() + index, self.data())
    }
}

pub trait MatrixIndex {
    fn bound(&self) -> (Option<usize>, Option<usize>);
}

impl MatrixIndex for usize {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(*self), Some(*self + 1))
    }
}

impl MatrixIndex for Range<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(self.start), Some(self.end))
    }
}

impl MatrixIndex for RangeFrom<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(self.start), None)
    }
}

impl MatrixIndex for RangeFull {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, None)
    }
}

impl MatrixIndex for RangeInclusive<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(*self.start()), Some(*self.end() + 1))
    }
}

impl MatrixIndex for RangeTo<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, Some(self.end))
    }
}

impl MatrixIndex for RangeToInclusive<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, Some(self.end + 1))
    }
}

impl Matrix {
    pub fn submatrix(&self, row_index: impl MatrixIndex, col_index: impl MatrixIndex) -> Matrix {
        let (rows, cols) = self.shape();

        let (row_start, row_end) = row_index.bound();
        let row_start = row_start.unwrap_or(0);
        let row_end = row_end.unwrap_or(rows);

        assert!(
            row_end <= rows,
            "Row index {}..{} out of range for matrix {}x{}",
            row_start,
            row_end,
            rows,
            cols
        );

        assert!(
            row_start < row_end,
            "Empty row index {}..{} for matrix {}x{}",
            row_start,
            row_end,
            rows,
            cols
        );

        let (col_start, col_end) = col_index.bound();
        let col_start = col_start.unwrap_or(0);
        let col_end = col_end.unwrap_or(cols);

        assert!(
            col_end <= cols,
            "Col index {}..{} out of range for matrix {}x{}",
            col_start,
            col_end,
            rows,
            cols
        );

        assert!(
            col_start < col_end,
            "Empty col index {}..{} for matrix {}x{}",
            col_start,
            col_end,
            rows,
            cols
        );

        let stride = self.stride();
        let offset = self.offset();
        Matrix::with_data(
            row_end - row_start,
            col_end - col_start,
            stride,
            offset + row_start * stride + col_start,
            self.data(),
        )
    }
}
