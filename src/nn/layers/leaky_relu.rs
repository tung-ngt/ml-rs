use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad},
    tensor::Tensor,
};

use super::DynGrad;
pub struct LeakyReLU<const INPUT_DIMENSIONS: usize> {
    input: Option<Tensor<INPUT_DIMENSIONS>>,
    negative_slope: f32,
}

impl<const INPUT_DIMENSIONS: usize> Default for LeakyReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self {
            input: None,
            negative_slope: 0.01,
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> LeakyReLU<INPUT_DIMENSIONS> {
    pub fn new(negative_slope: f32) -> Self {
        Self {
            input: None,
            negative_slope,
        }
    }

    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.negative_slope * x
        }
    }

    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            self.negative_slope
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Forward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for LeakyReLU<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<INPUT_DIMENSIONS> {
        self.input = Some(input.clone());
        input.apply(|x| self.leaky_relu(x))
    }
}

pub struct LeakyReLUGrad<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
}

impl<const INPUT_DIMENSIONS: usize> InputGrad<INPUT_DIMENSIONS>
    for LeakyReLUGrad<INPUT_DIMENSIONS>
{
    fn input(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for LeakyReLU<INPUT_DIMENSIONS>
{
    type Grad = LeakyReLUGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self::Grad {
        Self::Grad {
            input: self
                .input
                .as_ref()
                .expect("havent forward")
                .apply(|x| self.derivative(x))
                .mul_elem(next_grad),
        }
    }
}

impl DynLayer for LeakyReLU<2> {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        Forward::forward(self, input)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        DynGrad::LeakyReLU(Backward::backward(self, next_grad))
    }

    fn update(&mut self, _optimizer: &mut DynOptimizer, _grad: &DynGrad) {}
}
