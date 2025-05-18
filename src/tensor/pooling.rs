use super::Tensor;

pub fn pooling_output_shape(
    image_size: &[usize; 4],
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilation: (usize, usize),
) -> [usize; 4] {
    let &[b, h, w, c] = image_size;
    let h_out = (h - dilation.0 * (kernel_size.0 - 1) - 1) / strides.0 + 1;
    let w_out = (w - dilation.1 * (kernel_size.1 - 1) - 1) / strides.1 + 1;
    [b, h_out, w_out, c]
}

impl Tensor<4> {
    #[allow(non_snake_case)]
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> (Tensor<4>, Tensor<4>) {
        let &[B, H, W, C_in] = self.shape();
        let H_out = (H - dilations.0 * (kernel_size.0 - 1) - 1) / strides.0 + 1;
        let W_out = (W - dilations.1 * (kernel_size.1 - 1) - 1) / strides.1 + 1;
        let out_shape = [B, H_out, W_out, C_in];
        let data_strides = Self::get_strides(&out_shape);
        let no_elements = out_shape.iter().product();

        let mut data = Vec::with_capacity(no_elements);
        let mut indices = Vec::with_capacity(no_elements);

        for b in 0..B {
            for h_out in 0..H_out {
                for w_out in 0..W_out {
                    for c in 0..C_in {
                        let mut max = f32::NEG_INFINITY;
                        let mut index = 0.0;

                        for h_k in 0..kernel_size.0 {
                            for w_k in 0..kernel_size.1 {
                                let i = h_out * strides.0 + h_k * dilations.0;
                                let j = w_out * strides.1 + w_k * dilations.1;
                                let x = self[&[b, i, j, c]];

                                if x > max {
                                    max = x;
                                    index = (h_k * kernel_size.1 + w_k) as f32;
                                }
                            }
                        }

                        data.push(max);
                        indices.push(index);
                    }
                }
            }
        }

        (
            Tensor::with_data(&out_shape, &data_strides, 0, data.into()),
            Tensor::with_data(&out_shape, &data_strides, 0, indices.into()),
        )
    }
}

#[cfg(test)]
mod pooling_tests {
    use crate::tensor;
    #[test]
    fn max_pool() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let b = tensor!(1, 2, 2, 1 => [
            4.0, 5.0,
            6.0, 7.0
        ]);

        let idx = tensor!(1, 2, 2, 1 => [
            3.0, 0.0,
            2.0, 1.0
        ]);

        let (c, indices) = a.max_pool2d((2, 2), (2, 2), (1, 1));

        assert!(c == b);
        assert!(idx == indices);
    }

    #[test]
    fn max_pool_misalign() {
        let a = tensor!(1, 4, 5, 1 => [
            1.0, 2.0, 5.0, 3.0, 10.0,
            3.0, 4.0, 2.0, 1.0, 10.0,
            3.0, 2.0, 3.0, 7.0, 10.0,
            6.0, 1.0, 2.0, 1.0, 10.0
        ]);

        let b = tensor!(1, 2, 2, 1 => [
            4.0, 5.0,
            6.0, 7.0
        ]);

        let idx = tensor!(1, 2, 2, 1 => [
            3.0, 0.0,
            2.0, 1.0
        ]);

        let (c, indices) = a.max_pool2d((2, 2), (2, 2), (1, 1));

        assert!(c == b);
        assert!(idx == indices);
    }
}
