use std::ops::RangeBounds;

use super::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn flatten<const OUTPUT_DIMENSIONS: usize>(
        &self,
        flatten_dims: impl RangeBounds<usize>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let shape = self.shape();
        let mut new_shape = [1; OUTPUT_DIMENSIONS];

        let mut n = OUTPUT_DIMENSIONS;
        for (i, s) in self.shape().iter().enumerate().rev() {
            if !flatten_dims.contains(&i) || i == NO_DIMENSIONS - 1 {
                n -= 1;
            }
            new_shape[n] *= *s;
        }

        let mut data = Vec::with_capacity(self.no_elements());

        for i in 0..self.no_elements() {
            let multi_dim_index = Self::flat_to_nd_index(i, shape);
            data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(&new_shape, &Tensor::get_strides(&new_shape), 0, data.into())
    }

    pub fn reshape<const OUTPUT_DIMENSIONS: usize>(
        &self,
        new_shape: &[usize; OUTPUT_DIMENSIONS],
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let mut data = Vec::with_capacity(self.no_elements());

        let shape = self.shape();
        for i in 0..self.no_elements() {
            let multi_dim_index = Self::flat_to_nd_index(i, shape);
            data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(new_shape, &Tensor::get_strides(new_shape), 0, data.into())
    }
}

#[cfg(test)]
mod reshape_test {
    use crate::tensor;
    use crate::tensor::Tensor;

    #[test]
    fn flatten() {
        let a = tensor!(2, 3, 3 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);

        let b: Tensor<2> = a.flatten(0..1);
        let c = tensor!(6, 3 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);
        assert!(b == c);

        let b: Tensor<2> = a.flatten(1..2);
        let c = tensor!(2, 9 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);
        assert!(b == c);

        let b: Tensor<1> = a.flatten(..);
        let c = tensor!(18 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);
        assert!(b == c);
    }

    #[test]
    fn reshape() {
        let a = tensor!(18 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);

        let b = tensor!(2, 3, 3 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);

        let c = a.reshape(&[2, 3, 3]);

        assert!(b == c);
    }
}
