use super::Tensor;

impl<const NO_DIMENSIONS: usize> Tensor<NO_DIMENSIONS> {
    pub fn reverse_all(&self) -> Tensor<NO_DIMENSIONS> {
        let mut reverse_data = Vec::with_capacity(self.no_elements());
        let strides = self.strides();
        let mut multi_dim_index: [usize; NO_DIMENSIONS];

        for flat_index in (0..self.no_elements()).rev() {
            multi_dim_index = Self::flat_to_nd_index(flat_index, strides);
            reverse_data.push(self[&multi_dim_index]);
        }

        Tensor::with_data(
            self.shape(),
            &Self::get_strides(self.shape()),
            0,
            reverse_data.into(),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;
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
