use super::Tensor;

pub fn dilate_output_shape(image_shape: &[usize; 4], dilations: (usize, usize)) -> [usize; 4] {
    let &[b, h, w, c] = image_shape;
    let h_out = dilations.0 * (h - 1) + 1;
    let w_out = dilations.1 * (w - 1) + 1;
    [b, h_out, w_out, c]
}
impl Tensor<4> {
    #![allow(non_snake_case)]
    pub fn dilate2d(&self, dilations: (usize, usize)) -> Tensor<4> {
        let dilate_shape = dilate_output_shape(self.shape(), dilations);
        let dilate_strides = Tensor::get_strides(&dilate_shape);
        let mut data = vec![0.0; dilate_shape.iter().product()];

        let &[B, H, W, C] = self.shape();

        for b in 0..B {
            for h in 0..H {
                for w in 0..W {
                    for c in 0..C {
                        let i = h * dilations.0;
                        let j = w * dilations.1;

                        data[b * dilate_strides[0]
                            + i * dilate_strides[1]
                            + j * dilate_strides[2]
                            + c * dilate_strides[3]] = self[&[b, h, w, c]];
                    }
                }
            }
        }

        Tensor::with_data(&dilate_shape, &dilate_strides, 0, data.into())
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;
    #[test]
    fn dilate() {
        let a = tensor!(1, 3, 3, 1 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]);

        let b = tensor!(1, 7, 7, 1 => [
            1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0
        ]);

        let c = a.dilate2d((3, 3));

        assert!(b == c);
    }
}
