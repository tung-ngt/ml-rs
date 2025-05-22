use super::TensorIndex;
use crate::tensor::Tensor;

impl Tensor<2> {
    pub fn row(&self, index: usize) -> Tensor<2> {
        let &[rows, cols] = self.shape();
        assert!(
            index < rows,
            "Row index {} out of range for matrix {}x{}",
            index,
            rows,
            cols
        );

        let strides = self.strides();
        Tensor::matrix_with_data(
            1,
            cols,
            strides,
            self.offset() + index * strides[0],
            self.data(),
        )
    }

    pub fn col(&self, index: usize) -> Tensor<2> {
        let &[rows, cols] = self.shape();
        assert!(
            index < cols,
            "Col index {} out of range for matrix {}x{}",
            index,
            rows,
            cols
        );
        Tensor::matrix_with_data(rows, 1, self.strides(), self.offset() + index, self.data())
    }

    pub fn diag(&self) -> Tensor<1> {
        let &[rows, cols] = self.shape();
        assert!(
            rows == cols,
            "Should be square matrix to get diagonal axis. Got matrix {}x{}",
            rows,
            cols
        );
        let new_stride = self.strides()[0] + 1;
        Tensor::vector_with_data(rows, new_stride, self.offset(), self.data())
    }
}

impl Tensor<2> {
    pub fn submatrix(&self, row_index: impl TensorIndex, col_index: impl TensorIndex) -> Tensor<2> {
        let &[rows, cols] = self.shape();

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

        let strides = self.strides();
        let offset = self.offset();
        Tensor::matrix_with_data(
            row_end - row_start,
            col_end - col_start,
            strides,
            offset + row_start * strides[0] + col_start,
            self.data(),
        )
    }
}
