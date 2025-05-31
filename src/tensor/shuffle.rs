use super::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn shuffle_batch(&self, shuffled_index: &[usize]) -> Tensor<NO_DIMENSIONS> {
        let shape = self.shape();
        let strides = Tensor::get_strides(shape);
        let no_elements = self.no_elements();
        let mut data = Vec::with_capacity(no_elements);

        for flat_index in 0..no_elements {
            let mut multi_dim_index = Tensor::flat_to_nd_index(flat_index, shape);
            multi_dim_index[0] = shuffled_index[multi_dim_index[0]];
            data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(shape, &strides, 0, data.into())
    }
}
