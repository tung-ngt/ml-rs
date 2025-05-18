use super::Tensor;

pub fn conv_output_shape(
    images_size: &[usize; 4],
    kernels_size: &[usize; 4],
    strides: (usize, usize),
) -> [usize; 4] {
    let &[b, h, w, _c_in] = images_size;
    let &[c_out, k_h, k_w, _] = kernels_size;
    let new_h = (h - k_h) / strides.0 + 1;
    let new_w = (w - k_w) / strides.1 + 1;
    [b, new_w, new_h, c_out]
}

pub fn conv_unused_inputs(
    image_size: (usize, usize),
    kernel_size: (usize, usize),
    strides: (usize, usize),
) -> (usize, usize) {
    let [_, h_out, w_out, _] = conv_output_shape(
        &[1, image_size.0, image_size.1, 1],
        &[1, kernel_size.0, kernel_size.1, 1],
        strides,
    );
    let unused_h = image_size.0 - (h_out - 1) * strides.0 - kernel_size.0;
    let unused_w = image_size.1 - (w_out - 1) * strides.1 - kernel_size.1;
    (unused_h, unused_w)
}

impl Tensor<4> {
    /// Apply 2D convolution on a matrix
    /// Note that this function actually just do cross correlation
    /// Note that this operation does not add padding
    ///
    /// Expect the matrix to be the form (BxHxWxC)
    /// Expect the kernel to be the form (C_outxH_KxW_KxC)
    /// Will produce a matrix with the form (BxH_outxW_outxC_out)
    /// H_out = (H - H_k)/2 + 1
    /// W_out = (W - W_k)/2 + 1
    #[allow(non_snake_case)]
    pub fn conv2d(&self, kernels: &Tensor<4>, strides: (usize, usize)) -> Tensor<4> {
        assert!(
            strides.0 > 0 && strides.1 > 0,
            "stride must be > 0 got: {:?}",
            strides
        );

        let &[_, H_k, W_k, C] = kernels.shape();
        let &[_, _, _, C_in] = self.shape();
        assert!(
            C_in == C,
            "Image and kernel channels do not match image channels: {}, kernel channels: {}",
            C_in,
            C
        );

        let out_shape = conv_output_shape(self.shape(), kernels.shape(), strides);
        let [B, H_out, W_out, C_out] = out_shape;

        let mut data = Vec::with_capacity(out_shape.iter().product());

        for b in 0..B {
            for h_out in 0..H_out {
                for w_out in 0..W_out {
                    for c_out in 0..C_out {
                        let mut pixel_sum = 0.0;

                        for h_k in 0..H_k {
                            for w_k in 0..W_k {
                                for c_in in 0..C_in {
                                    pixel_sum += self[&[
                                        b,
                                        h_out * strides.0 + h_k,
                                        w_out * strides.1 + w_k,
                                        c_in,
                                    ]] * kernels[&[c_out, h_k, w_k, c_in]];
                                }
                            }
                        }

                        data.push(pixel_sum);
                    }
                }
            }
        }

        Tensor::with_data(&out_shape, &Tensor::get_strides(&out_shape), 0, data.into())
    }
}

#[cfg(test)]
mod conv_tests {
    use crate::tensor;
    use crate::tensor::{
        pad::{pad2d_same_size, PaddingType},
        Tensor,
    };

    #[test]
    fn conv() {
        let a1 = Tensor::<3>::filled(&[5, 5, 1], 1.0);
        let a2 = Tensor::<3>::filled(&[5, 5, 1], 2.0);
        let a = Tensor::stack(&[a1, a2]);
        let kernel = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let pad_a = a.pad2d(pad2d_same_size((5, 5), (3, 3), (1, 1)), PaddingType::Zero);
        let b = pad_a.conv2d(&kernel, (1, 1));

        let c = tensor!(2, 5, 5, 1 => [
            4.0, 6.0, 6.0, 6.0, 4.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            4.0, 6.0, 6.0, 6.0, 4.0,

             8.0, 12.0, 12.0, 12.0,  8.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
             8.0, 12.0, 12.0, 12.0,  8.0
        ]);

        assert!(b == c);
    }
}
