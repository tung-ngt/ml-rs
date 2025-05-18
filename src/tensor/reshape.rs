use super::Tensor;

pub fn flatten_output_shape<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>(
    input_shape: &[usize; INPUT_DIMENSIONS],
    start: Option<usize>,
    stop: Option<usize>,
) -> [usize; OUTPUT_DIMENSIONS] {
    let mut new_shape = [1; OUTPUT_DIMENSIONS];

    let start = start.unwrap_or(0);
    let stop = stop.unwrap_or(INPUT_DIMENSIONS);

    let flatten_dims = start..stop;

    let mut n = OUTPUT_DIMENSIONS;
    for (i, s) in input_shape.iter().enumerate().rev() {
        if !flatten_dims.contains(&i) || i == INPUT_DIMENSIONS - 1 {
            n -= 1;
        }
        new_shape[n] *= *s;
    }
    new_shape
}

impl<const INPUT_DIMENSIONS: usize> Tensor<INPUT_DIMENSIONS> {
    pub fn flatten<const OUTPUT_DIMENSIONS: usize>(
        &self,
        start: Option<usize>,
        stop: Option<usize>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let shape = self.shape();
        let new_shape = flatten_output_shape(self.shape(), start, stop);
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

        let b: Tensor<2> = a.flatten(None, Some(1));
        let c = tensor!(6, 3 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);
        assert!(b == c);

        let b: Tensor<2> = a.flatten(Some(1), Some(2));
        let c = tensor!(2, 9 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,

            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0
        ]);
        assert!(b == c);

        let b: Tensor<1> = a.flatten(None, None);
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
