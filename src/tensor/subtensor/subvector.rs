use super::TensorIndex;
use crate::tensor::Tensor;

impl Tensor<1> {
    pub fn subvector(&self, index: impl TensorIndex) -> Tensor<1> {
        let &[size] = self.shape();

        let (start, end) = index.bound();
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(size);

        assert!(
            end <= size,
            "index {}..{} out of range for vector ({},)",
            start,
            end,
            size,
        );

        assert!(
            start < end,
            "Empty index {}..{} out of range for vector ({},)",
            start,
            end,
            size,
        );

        let strides = self.strides();
        let offset = self.offset();

        Tensor::vector_with_data(
            end - start,
            strides[0],
            offset + start * strides[0],
            self.data(),
        )
    }
}

#[cfg(test)]
mod subvector_tests {
    use crate::tensor;
    #[test]
    fn subvector() {
        let a1 = tensor!(5 => [1.0, 2.0, 3.0, 4.0, 5.0]);

        let a2 = tensor!(1 => [3.0]);
        let a3 = tensor!(2 => [2.0, 3.0]);
        let a4 = tensor!(2 => [4.0, 5.0]);
        let a5 = tensor!(5 => [1.0, 2.0, 3.0, 4.0, 5.0]);
        let a6 = tensor!(3 => [2.0, 3.0, 4.0]);
        let a7 = tensor!(3 => [1.0, 2.0, 3.0]);
        let a8 = tensor!(4 => [1.0, 2.0, 3.0, 4.0]);

        let b2 = a1.subvector(2);
        let b3 = a1.subvector(1..3);
        let b4 = a1.subvector(3..);
        let b5 = a1.subvector(..);
        let b6 = a1.subvector(1..=3);
        let b7 = a1.subvector(..3);
        let b8 = a1.subvector(..=3);

        assert!(b2 == a2);
        assert!(b3 == a3);
        assert!(b4 == a4);
        assert!(b5 == a5);
        assert!(b6 == a6);
        assert!(b7 == a7);
        assert!(b8 == a8);
    }
}
