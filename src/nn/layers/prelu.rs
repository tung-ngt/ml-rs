use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad, Optimizer, Update},
    tensor::Tensor,
};

use super::DynGrad;
pub struct PReLU<const INPUT_DIMENSIONS: usize> {
    input: Option<Tensor<INPUT_DIMENSIONS>>,
    negative_slope: Tensor<1>,
}

impl<const INPUT_DIMENSIONS: usize> Default for PReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self {
            input: None,
            negative_slope: Tensor::vector_filled(1, 0.01),
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> PReLU<INPUT_DIMENSIONS> {
    pub fn new(negative_slope: f32) -> Self {
        Self {
            input: None,
            negative_slope: Tensor::vector_filled(1, negative_slope),
        }
    }

    fn prelu(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.negative_slope[0] * x
        }
    }

    fn derivative_wrt_input(&self, x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            self.negative_slope[0]
        }
    }

    fn derivative_wrt_slope(&self, x: f32) -> f32 {
        if x > 0.0 {
            0.0
        } else {
            x
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Forward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for PReLU<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<INPUT_DIMENSIONS> {
        self.input = Some(input.clone());
        input.apply(|x| self.prelu(x))
    }
}

pub struct PReLUGrad<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
    negative_slope: Tensor<1>,
}

impl<const INPUT_DIMENSIONS: usize> InputGrad<INPUT_DIMENSIONS> for PReLUGrad<INPUT_DIMENSIONS> {
    fn input(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for PReLU<INPUT_DIMENSIONS>
{
    type Grad = PReLUGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self::Grad {
        Self::Grad {
            input: self
                .input
                .as_ref()
                .expect("havent forward")
                .apply(|x| self.derivative_wrt_input(x))
                .mul_elem(next_grad),
            negative_slope: self.negative_slope.apply(|x| self.derivative_wrt_slope(x)),
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for PReLU<INPUT_DIMENSIONS> {
    type Grad = PReLUGrad<INPUT_DIMENSIONS>;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        optimizer.step(&mut self.negative_slope, &grad.negative_slope);
    }
}

impl DynLayer for PReLU<2> {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        Forward::forward(self, input)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        DynGrad::PReLU(Backward::backward(self, next_grad))
    }

    fn update(&mut self, optimizer: &mut DynOptimizer, grad: &DynGrad) {
        let DynGrad::PReLU(grad) = grad else {
            return;
        };

        optimizer.step(&mut self.negative_slope, &grad.negative_slope);
    }
}
