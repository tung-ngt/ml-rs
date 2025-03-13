use super::TensorIndex;
use crate::tensor::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn subtensor(&self, indices: &[impl TensorIndex; NO_DIMENSIONS]) -> Tensor<NO_DIMENSIONS> {
        let shape = self.shape();

        let mut indices_bounds = [(0, 0); NO_DIMENSIONS];
        let mut new_shape = [0; NO_DIMENSIONS];
        for i in 0..shape.len() {
            let (start, end) = indices[i].bound();
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(shape[i]);

            assert!(
                end <= shape[i],
                "Index for dim {}: {}..{} out of range for tensor {:?}",
                i,
                start,
                end,
                shape
            );

            assert!(
                start < end,
                "Empty index for dim {}: {}..{} for tensor {:?}",
                i,
                start,
                end,
                shape
            );

            indices_bounds[i].0 = start;
            indices_bounds[i].1 = end;
            new_shape[i] = end - start;
        }

        let strides = self.strides();
        let offset = self.offset();

        let new_offset = strides
            .iter()
            .zip(indices_bounds)
            .fold(offset, |acc, (&stride, bound)| acc + stride * bound.0);
        Tensor::with_data(&new_shape, strides, new_offset, self.data())
    }
}
