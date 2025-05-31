use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad},
    tensor::{
        pad::{pad2d_output_size, PaddingSize, PaddingType},
        Tensor,
    },
};

use super::DynGrad;

pub struct Pad2D {
    input_shape: Option<[usize; 4]>,
    pad_size: (PaddingSize, PaddingSize),
    pad_type: PaddingType,
}

impl Pad2D {
    pub fn new(pad_size: (PaddingSize, PaddingSize), pad_type: PaddingType) -> Self {
        Self {
            input_shape: None,
            pad_size,
            pad_type,
        }
    }

    pub fn with_shape(
        input_shape: &[usize; 4],
        pad_size: (PaddingSize, PaddingSize),
        pad_type: PaddingType,
    ) -> Self {
        Self {
            input_shape: Some(*input_shape),
            pad_size,
            pad_type,
        }
    }
}

impl Forward<4, 4> for Pad2D {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        input.pad2d(self.pad_size.clone(), self.pad_type.clone())
    }
}

pub struct Pad2DGrad {
    input: Tensor<4>,
}

impl InputGrad<4> for Pad2DGrad {
    fn input(&self) -> Tensor<4> {
        self.input.clone()
    }
}

impl Backward<4, 4> for Pad2D {
    type Grad = Pad2DGrad;
    fn backward(&self, next_grad: &Tensor<4>) -> Self::Grad {
        let &[b, h, w, c] = next_grad.shape();
        let (top, bottom) = match self.pad_size.0 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(t, b) => (t, b),
        };

        let (left, right) = match self.pad_size.1 {
            PaddingSize::Same(p) => (p, p),
            PaddingSize::Diff(l, r) => (l, r),
        };
        Pad2DGrad {
            input: next_grad.subtensor(&[0..b, top..h - bottom, left..w - right, 0..c]),
        }
    }
}

impl DynLayer for Pad2D {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let input = input.reshape(
            self.input_shape
                .as_ref()
                .expect("must give image shape for dyn conv forward"),
        );
        Forward::forward(self, &input).flatten(Some(1), None)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        let image_shape = self
            .input_shape
            .as_ref()
            .expect("must give image shape for dyn conv backward");
        let output_shape = pad2d_output_size(image_shape, self.pad_size.clone());
        let next_grad = next_grad.reshape(&output_shape);
        DynGrad::Pad2D(Backward::backward(self, &next_grad))
    }

    fn update(&mut self, _optimizer: &mut DynOptimizer, _grad: &DynGrad) {}
}

#[cfg(test)]
mod pad_test {
    use crate::nn::{Backward, Forward, InputGrad};
    use crate::tensor;
    use crate::tensor::pad::PaddingSize;

    use super::Pad2D;
    #[test]
    fn forward_backward() {
        let a = tensor!(1, 3, 3, 1 => [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]);

        let b = tensor!(1, 5, 7, 1 => [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ]);

        let mut pad = Pad2D::new(
            (PaddingSize::Same(1), PaddingSize::Same(2)),
            tensor::pad::PaddingType::Zero,
        );

        let c = pad.forward(&a);

        assert!(b == c, "expected {}\n got {}", b.squeeze(0), c.squeeze(0));

        let next_grad = tensor!(1, 5, 7, 1 => [
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
        ]);

        let expected_grad = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let grad = pad.backward(&next_grad);

        assert!(expected_grad == grad.input());
    }
}
