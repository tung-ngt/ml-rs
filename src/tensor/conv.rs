use std::usize;

use super::Tensor;

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
    pub fn conv2d(&self, kernels: &Tensor<4>, strides: (usize, usize)) -> Tensor<4> {
        assert!(
            strides.0 > 0 && strides.1 > 0,
            "stride must be > 0 got: {:?}",
            strides
        );

        let &[C_out, H_k, W_k, C] = kernels.shape();
        assert!(
            H_k % 2 == 1 && W_k % 2 == 1,
            "We dont support even kernal shape, {}x{}x{}x{}",
            C_out,
            H_k,
            W_k,
            C
        );

        let &[B, H, W, C_in] = self.shape();
        assert!(
            C_in == C,
            "Image and kernel channels do not match image channels: {}, kernel channels: {}",
            C_in,
            C
        );

        let H_out = (H - H_k) / strides.0 + 1;
        let W_out = (W - W_k) / strides.1 + 1;
        let out_shape = [B, H_out, W_out, C_out];

        let mut data = Vec::with_capacity(out_shape.iter().product());

        for b in 0..B {
            for c_out in 0..C_out {
                for h_out in 0..H_out {
                    for w_out in 0..W_out {
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

    // Add padding to the matrix
    // expect padding_sizes (height_padding, width_padding)
    // the padding_size is the size for one side
    pub fn pad2d(
        &self,
        padding_sizes: (PaddingSize, PaddingSize),
        padding_type: PaddingType,
    ) -> Tensor<4> {
        let value = match padding_type {
            PaddingType::Zero => 0.0,
            PaddingType::Const(v) => v,
            PaddingType::Replicate => unimplemented!("Havent implemented replicate padding"),
        };

        let &[b, h, w, c] = self.shape();
        let (top, bottom) = match padding_sizes.0 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(t, b) => (t, b),
        };

        let (left, right) = match padding_sizes.0 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(l, r) => (l, r),
        };

        let new_shape = [b, h + top + bottom, w + left + right, c];
        let mut data = vec![value; new_shape.iter().product()];

        for i in 0..b {
            for j in 0..h {
                for k in 0..w {
                    for l in 0..c {
                        let pad_j = j + top;
                        let pad_k = k + left;
                        let index = i * new_shape[1..].iter().product::<usize>()
                            + pad_j * new_shape[2..].iter().product::<usize>()
                            + pad_k * new_shape[3]
                            + l;
                        data[index] = self[&[i, j, k, l]];
                    }
                }
            }
        }

        Tensor::with_data(&new_shape, &Tensor::get_strides(&new_shape), 0, data.into())
    }
}

pub enum PaddingType {
    Zero,
    Const(f32),
    Replicate,
}

pub enum PaddingSize {
    Same(usize),
    Diff(usize, usize),
}

#[cfg(test)]
mod test {
    use crate::tensor;
    use crate::tensor::conv::PaddingSize;
    use crate::tensor::utils::pad2d_same_size;
    use crate::tensor::Tensor;

    use super::PaddingType;

    #[test]
    fn pad() {
        let a = Tensor::<4>::filled(&[1, 3, 3, 3], 1.0);
        let pad_a = a.pad2d(
            (PaddingSize::Same(2), PaddingSize::Same(2)),
            PaddingType::Zero,
        );

        let b = tensor!(1, 7, 7, 3 => [
            0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
            0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0
        ]);

        assert!(pad_a == b);
    }

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
