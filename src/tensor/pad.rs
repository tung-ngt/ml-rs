use super::Tensor;

#[derive(Clone)]
pub enum PaddingType {
    Zero,
    Const(f32),
    Replicate,
}

#[derive(Clone)]
pub enum PaddingSize {
    Same(usize),
    Diff(usize, usize),
}

pub fn pad2d_same_size(
    image_size: (usize, usize),
    kernel_size: (usize, usize),
    strides: (usize, usize),
) -> (PaddingSize, PaddingSize) {
    let height_padding = image_size.0 * (strides.0 - 1) + kernel_size.0 - strides.0;
    let height_padding = if height_padding % 2 == 0 {
        PaddingSize::Same(height_padding / 2)
    } else {
        PaddingSize::Diff(height_padding / 2, height_padding / 2 + 1)
    };

    let width_padding = image_size.1 * (strides.1 - 1) + kernel_size.1 - strides.1;
    let width_padding = if width_padding % 2 == 0 {
        PaddingSize::Same(width_padding / 2)
    } else {
        PaddingSize::Diff(width_padding / 2, width_padding / 2 + 1)
    };

    (height_padding, width_padding)
}

pub fn pad2d_full_size(kernel_size: (usize, usize)) -> (PaddingSize, PaddingSize) {
    let height_padding = PaddingSize::Same(kernel_size.0 - 1);
    let width_padding = PaddingSize::Same(kernel_size.1 - 1);

    (height_padding, width_padding)
}

pub fn pad2d_output_size(
    input_shape: &[usize; 4],
    padding_sizes: (PaddingSize, PaddingSize),
) -> [usize; 4] {
    let &[b, h, w, c] = input_shape;
    let (top, bottom) = match padding_sizes.0 {
        PaddingSize::Same(p) => (p, p),
        PaddingSize::Diff(t, b) => (t, b),
    };

    let (left, right) = match padding_sizes.1 {
        PaddingSize::Same(p) => (p, p),
        PaddingSize::Diff(l, r) => (l, r),
    };

    [b, h + top + bottom, w + left + right, c]
}

impl Tensor<4> {
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
        let (top, _) = match padding_sizes.0 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(t, b) => (t, b),
        };

        let (left, _) = match padding_sizes.1 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(l, r) => (l, r),
        };

        let new_shape = pad2d_output_size(self.shape(), padding_sizes);

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

#[cfg(test)]
mod pad_tests {
    use super::PaddingSize;
    use crate::tensor;
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
}
