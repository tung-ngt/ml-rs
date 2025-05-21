use super::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    /// Unfinised stacking
    pub fn stack<const OUTPUT_DIMENSIONS: usize>(
        tensors: &[Tensor<NO_DIMENSIONS>],
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let no_tensors = tensors.iter().len();
        let shape = tensors[0].shape();
        let no_elements = tensors[0].no_elements();

        let mut new_data = Vec::with_capacity(no_tensors * no_elements);
        let mut new_shape = [0; OUTPUT_DIMENSIONS];
        new_shape[0] = no_tensors;

        for (i, s) in shape.iter().enumerate() {
            new_shape[i + 1] = *s;
        }

        for tensor in tensors {
            for flat_index in 0..no_elements {
                let multi_dim_index = Tensor::flat_to_nd_index(flat_index, shape);
                new_data.push(tensor[&multi_dim_index]);
            }
        }
        Tensor::with_data(
            &new_shape,
            &Tensor::get_strides(&new_shape),
            0,
            new_data.into(),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;
    use crate::tensor::Tensor;

    #[test]
    fn stack() {
        let a = tensor!(3, 3, 2 => [
            0.0,0.0, 0.0,0.0, 0.0,0.0,
            0.0,0.0, 0.0,0.0, 0.0,0.0,
            0.0,0.0, 0.0,0.0, 0.0,0.0
        ]);

        let b = tensor!(3, 3, 2 => [
            1.0,1.0, 1.0,1.0, 1.0,1.0,
            1.0,1.0, 1.0,1.0, 1.0,1.0,
            1.0,1.0, 1.0,1.0, 1.0,1.0
        ]);

        let c = Tensor::stack(&[a, b]);

        let d = tensor!(2, 3, 3, 2 => [
            0.0,0.0, 0.0,0.0, 0.0,0.0,
            0.0,0.0, 0.0,0.0, 0.0,0.0,
            0.0,0.0, 0.0,0.0, 0.0,0.0,

            1.0,1.0, 1.0,1.0, 1.0,1.0,
            1.0,1.0, 1.0,1.0, 1.0,1.0,
            1.0,1.0, 1.0,1.0, 1.0,1.0
        ]);

        assert!(c == d);
    }
}
