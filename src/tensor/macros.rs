#[macro_export]
macro_rules! tensor {
    ($($shape:expr),+ => [$([$($a:tt),+]),+ ]) => {{
        tensor!($($shape),+ => [$($($a),+),+])
    }};

    ($($shape:expr),+ => [$([$($a:expr),+]),+ ]) => {{
        tensor!($($shape),+ => [$($($a),+),+])
    }};

    ($($shape:expr),+ => [$($a:expr),+] ) => {{
        let shape: [usize; tensor!(@count_ele; $($shape),+)] = [$($shape),+];
        let strides = $crate::tensor::Tensor::get_strides(&shape);
        let data: Vec<f32> = vec![$($a),+];
        const {
            assert!(
                tensor!(@count_ele_dims; $($shape),+) == tensor!(@count_ele; $($a),+),
                "No elements does not match shape"
            );
        };
        $crate::tensor::Tensor::with_data(&shape, &strides, 0, std::sync::Arc::from(data))
    }};

    (@count_ele; $($ele:expr),+) => (0usize $(+ { let _ = $ele; 1})+);
    (@count_ele_dims; $($dim:expr),+) => (1usize $(* $dim)+);
}

#[cfg(test)]
mod tensor_macro_tests {
    #[test]
    fn nested_3() {
        let a = tensor!(2, 3, 2 => [
            [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
            [[4.0, -4.0], [5.0, -5.0], [6.0, -6.0]]
        ]);
        let data = [
            1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0,
        ];

        assert!(a.shape() == &[2, 3, 2]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..1 {
                    assert!(a[&[i, j, k]] == data[i * 6 + j * 2 + k]);
                }
            }
        }
    }

    #[test]
    fn vec() {
        let a = tensor!(5 => [1.0, 2.0, 3.0, 4.0, 5.0]);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(a.shape() == &[5]);
        for i in 0..5 {
            assert!(a[i] == data[i]);
        }
    }

    #[test]
    fn matrix_flatten() {
        let a = tensor!(2, 3 => [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        ]);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert!(a.shape() == &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert!(a[(i, j)] == data[i * 3 + j]);
            }
        }
    }

    #[test]
    fn matrix() {
        let a = tensor!(2, 3 => [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert!(a.shape() == &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert!(a[(i, j)] == data[i * 3 + j]);
            }
        }
    }

    #[test]
    fn tensor_4d() {
        let a = tensor!(2, 3, 5, 5 => [
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0]
                ],

                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0]
                ],

                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0]
                ]

            ],

            [
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                    [-5.0, -5.0, -5.0, -5.0, -5.0]
                ],

                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                    [-5.0, -5.0, -5.0, -5.0, -5.0]
                ],

                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                    [-5.0, -5.0, -5.0, -5.0, -5.0]
                ]

            ]
        ]);
        let mut data = vec![0f32; 2 * 3 * 5 * 5];

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..5 {
                        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                        data[i * 75 + j * 25 + k * 5 + l] = sign * (k + 1) as f32;
                    }
                }
            }
        }

        assert!(a.shape() == &[2, 3, 5, 5]);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..5 {
                        assert!(data[i * 75 + j * 25 + k * 5 + l] == a[&[i, j, k, l]]);
                    }
                }
            }
        }
    }
}
