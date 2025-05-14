use super::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn reverse(&self, reverse_dims: &[usize]) -> Tensor<NO_DIMENSIONS> {
        let mut reverse_data = Vec::with_capacity(self.no_elements());
        let shape = self.shape();
        let mut multi_dim_index: [usize; NO_DIMENSIONS];
        for flat_index in 0..self.no_elements() {
            multi_dim_index = Self::flat_to_nd_index(flat_index, shape);

            for d in reverse_dims {
                multi_dim_index[*d] = shape[*d] - multi_dim_index[*d] - 1;
            }
            reverse_data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(shape, &Self::get_strides(shape), 0, reverse_data.into())
    }
    pub fn reverse_all(&self) -> Tensor<NO_DIMENSIONS> {
        let mut reverse_data = Vec::with_capacity(self.no_elements());
        let shape = self.shape();
        let mut multi_dim_index: [usize; NO_DIMENSIONS];

        for flat_index in (0..self.no_elements()).rev() {
            multi_dim_index = Self::flat_to_nd_index(flat_index, shape);
            reverse_data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(shape, &Self::get_strides(shape), 0, reverse_data.into())
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;

    #[test]
    fn reverse() {
        let a = tensor!(2, 3, 4 => [
             1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0,10.0,11.0,12.0,
            13.0,14.0,15.0,16.0, 17.0,18.0,19.0,20.0, 21.0,22.0,23.0,24.0
        ]);

        let a_reverse = a.reverse(&[0]);
        let b = tensor!(2, 3, 4 => [
            13.0,14.0,15.0,16.0, 17.0,18.0,19.0,20.0, 21.0,22.0,23.0,24.0,
             1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0,10.0,11.0,12.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[1]);
        let b = tensor!(2, 3, 4 => [
             9.0,10.0,11.0,12.0,  5.0, 6.0, 7.0, 8.0,  1.0, 2.0, 3.0, 4.0,
            21.0,22.0,23.0,24.0, 17.0,18.0,19.0,20.0, 13.0,14.0,15.0,16.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[2]);
        let b = tensor!(2, 3, 4 => [
             4.0, 3.0, 2.0, 1.0,  8.0, 7.0, 6.0, 5.0, 12.0,11.0,10.0, 9.0,
            16.0,15.0,14.0,13.0, 20.0,19.0,18.0,17.0, 24.0,23.0,22.0,21.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[0, 1]);
        let b = tensor!(2, 3, 4 => [
            21.0,22.0,23.0,24.0, 17.0,18.0,19.0,20.0, 13.0,14.0,15.0,16.0,
             9.0,10.0,11.0,12.0,  5.0, 6.0, 7.0, 8.0,  1.0, 2.0, 3.0, 4.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[1, 0]);
        let b = tensor!(2, 3, 4 => [
            21.0,22.0,23.0,24.0, 17.0,18.0,19.0,20.0, 13.0,14.0,15.0,16.0,
             9.0,10.0,11.0,12.0,  5.0, 6.0, 7.0, 8.0,  1.0, 2.0, 3.0, 4.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[2, 0]);
        let b = tensor!(2, 3, 4 => [
            16.0,15.0,14.0,13.0, 20.0,19.0,18.0,17.0, 24.0,23.0,22.0,21.0,
             4.0, 3.0, 2.0, 1.0,  8.0, 7.0, 6.0, 5.0, 12.0,11.0,10.0, 9.0
        ]);
        assert!(a_reverse == b);

        let a_reverse = a.reverse(&[2, 0, 1]);
        let b = tensor!(2, 3, 4 => [
            24.0,23.0,22.0,21.0, 20.0,19.0,18.0,17.0, 16.0,15.0,14.0,13.0,
            12.0,11.0,10.0, 9.0,  8.0, 7.0, 6.0, 5.0,  4.0, 3.0, 2.0, 1.0
        ]);
        assert!(a_reverse == b);
    }

    #[test]
    fn reverse_all() {
        let a = tensor!(2, 3, 4 => [
             1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0,10.0,11.0,12.0,
            13.0,14.0,15.0,16.0, 17.0,18.0,19.0,20.0, 21.0,22.0,23.0,24.0
        ]);

        let b = a.reverse_all();

        let c = tensor!(2, 3, 4 => [
            24.0,23.0,22.0,21.0, 20.0,19.0,18.0,17.0, 16.0,15.0,14.0,13.0,
            12.0,11.0,10.0, 9.0,  8.0, 7.0, 6.0, 5.0,  4.0, 3.0, 2.0, 1.0
        ]);

        assert!(b == c);
    }
}
