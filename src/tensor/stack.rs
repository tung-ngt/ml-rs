use super::Tensor;

impl Tensor<3> {
    /// Unfinised stacking
    pub fn stack(tensors: &[Tensor<3>]) -> Tensor<4> {
        let no_tensors = tensors.iter().len();
        let shape = tensors[0].shape();

        let mut new_data = Vec::with_capacity(no_tensors * tensors[0].no_elements());

        let new_shape = [no_tensors, shape[0], shape[1], shape[2]];

        for tensor in tensors {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        new_data.push(tensor[(i, j, k)]);
                    }
                }
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
