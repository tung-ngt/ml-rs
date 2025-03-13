use crate::tensor::Tensor;

impl Tensor<2> {
    pub fn squeeze(&self, removed_dim: usize) -> Tensor<1> {
        let &[rows, cols] = self.shape();
        let &[row_stride, col_stride] = self.strides();
        assert!(
            removed_dim < 2,
            "Cannot squeeze dim {} for matrix {}x{}",
            removed_dim,
            rows,
            cols
        );

        let new_shape = if removed_dim == 0 { cols } else { rows };

        let new_stride = if removed_dim == 0 {
            col_stride
        } else {
            row_stride
        };
        Tensor::vector_with_data(new_shape, new_stride, self.offset(), self.data())
    }
}

impl Tensor<3> {
    pub fn squeeze(&self, removed_dim: usize) -> Tensor<2> {
        let shape = self.shape();
        let strides = self.strides();

        assert!(
            removed_dim < 3,
            "Cannot squeeze dim {} for tensor {:?}",
            removed_dim,
            shape
        );

        let new_shape: Vec<_> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if i != removed_dim { Some(s) } else { None })
            .collect();

        let new_strides: Vec<_> = strides
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if i != removed_dim { Some(s) } else { None })
            .collect();

        Tensor::matrix_with_data(
            *new_shape[0],
            *new_shape[1],
            &[*new_strides[0], *new_strides[1]],
            self.offset(),
            self.data(),
        )
    }
}

impl Tensor<4> {
    pub fn squeeze(&self, removed_dim: usize) -> Tensor<3> {
        let shape = self.shape();
        let strides = self.strides();

        assert!(
            removed_dim < 4,
            "Cannot squeeze dim {} for tensor {:?}",
            removed_dim,
            shape
        );

        let new_shape: Vec<_> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if i != removed_dim { Some(s) } else { None })
            .collect();

        let new_shape = [*new_shape[0], *new_shape[1], *new_shape[2]];

        let new_strides: Vec<_> = strides
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if i != removed_dim { Some(s) } else { None })
            .collect();
        let new_strides = [*new_strides[0], *new_strides[1], *new_strides[2]];

        Tensor::with_data(&new_shape, &new_strides, self.offset(), self.data())
    }
}

#[cfg(test)]
mod tensor_squeeze_tests {
    use crate::tensor;

    #[test]
    fn matrix() {
        let a = tensor!(2 => [1.0, 2.0]);
        let b = tensor!(1, 2 => [[1.0, 2.0]]);
        let c = tensor!(2, 1 => [
            [1.0],
            [2.0]
        ]);

        assert!(b.squeeze(0) == a);
        assert!(c.squeeze(1) == a);
    }

    #[test]
    fn tensor_3d() {
        let a = tensor!(2, 2 => [
            [1.0, 2.0],
            [3.0, 4.0]
        ]);
        let b = tensor!(1, 2, 2 => [[
            [1.0, 2.0],
            [3.0, 4.0]
        ]]);

        let c = tensor!(2, 1, 2 => [
            [[1.0, 2.0]],
            [[3.0, 4.0]]
        ]);

        let d = tensor!(2, 2, 1 => [
            [[1.0], [2.0]],
            [[3.0], [4.0]]
        ]);

        assert!(b.squeeze(0) == a);
        assert!(c.squeeze(1) == a);
        assert!(d.squeeze(2) == a);
    }

    #[test]
    fn tensor_4d() {
        let a = tensor!(3, 2, 2 => [
            [
                [1.0, 1.0],
                [1.0, 1.0]
            ],
            [
                [2.0, 2.0],
                [2.0, 2.0]
            ],
            [
                [3.0, 3.0],
                [3.0, 3.0]
            ]
        ]);
        let b = tensor!(1, 3, 2, 2 => [[
            [
                [1.0, 1.0],
                [1.0, 1.0]
            ],
            [
                [2.0, 2.0],
                [2.0, 2.0]
            ],
            [
                [3.0, 3.0],
                [3.0, 3.0]
            ]
        ]]);
        let c = tensor!(3, 1, 2, 2 => [
            [[
                [1.0, 1.0],
                [1.0, 1.0]
            ]],
            [[
                [2.0, 2.0],
                [2.0, 2.0]
            ]],
            [[
                [3.0, 3.0],
                [3.0, 3.0]
            ]]
        ]);
        let d = tensor!(3, 2, 1, 2 => [
            [
                [[1.0, 1.0]],
                [[1.0, 1.0]]
            ],
            [
                [[2.0, 2.0]],
                [[2.0, 2.0]]
            ],
            [
                [[3.0, 3.0]],
                [[3.0, 3.0]]
            ]
        ]);

        let e = tensor!(3, 2, 2, 1 => [
            [
                [[1.0], [1.0]],
                [[1.0], [1.0]]
            ],
            [
                [[2.0], [2.0]],
                [[2.0], [2.0]]
            ],
            [
                [[3.0], [3.0]],
                [[3.0], [3.0]]
            ]
        ]);

        assert!(b.squeeze(0) == a);
        assert!(c.squeeze(1) == a);
        assert!(d.squeeze(2) == a);
        assert!(e.squeeze(3) == a);
    }
}
