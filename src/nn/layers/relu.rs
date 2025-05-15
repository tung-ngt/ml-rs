use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad},
    tensor::Tensor,
};

use super::DynGrad;
pub struct ReLU<const INPUT_DIMENSIONS: usize> {
    input: Option<Tensor<INPUT_DIMENSIONS>>,
}

impl<const INPUT_DIMENSIONS: usize> Default for ReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self { input: None }
    }
}

impl<const INPUT_DIMENSIONS: usize> ReLU<INPUT_DIMENSIONS> {
    fn relu(x: f32) -> f32 {
        0f32.max(x)
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Forward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for ReLU<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<INPUT_DIMENSIONS> {
        self.input = Some(input.clone());
        input.apply(Self::relu)
    }
}

pub struct ReLUGrad<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
}

impl<const INPUT_DIMENSIONS: usize> InputGrad<INPUT_DIMENSIONS> for ReLUGrad<INPUT_DIMENSIONS> {
    fn input(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for ReLU<INPUT_DIMENSIONS>
{
    type Grad = ReLUGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self::Grad {
        Self::Grad {
            input: self
                .input
                .as_ref()
                .expect("havent forward")
                .apply(Self::derivative)
                .mul_elem(next_grad),
        }
    }
}

impl DynLayer for ReLU<2> {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        Forward::forward(self, input)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        DynGrad::ReLU(Backward::backward(self, next_grad))
    }

    fn update(&mut self, _optimizer: &mut DynOptimizer, _grad: &DynGrad) {}
}
