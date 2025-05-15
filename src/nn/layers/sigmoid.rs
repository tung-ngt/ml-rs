use crate::nn::{Backward, Forward, InputGrad, Layer, Optimizer, Update};
use crate::tensor::Tensor;

pub struct Sigmoid<const INPUT_DIMENSIONS: usize> {
    input_sig: Option<Tensor<INPUT_DIMENSIONS>>,
}

impl<const INPUT_DIMENSIONS: usize> Default for Sigmoid<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self { input_sig: None }
    }
}

impl<const INPUT_DIMENSIONS: usize> Sigmoid<INPUT_DIMENSIONS> {
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl<const INPUT_DIMENSIONS: usize> Forward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for Sigmoid<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<INPUT_DIMENSIONS> {
        let output = input.apply(Self::sigmoid);
        self.input_sig = Some(output.clone());
        output
    }
}

pub struct SigmoidGrad<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
}

impl<const INPUT_DIMENSIONS: usize> InputGrad<INPUT_DIMENSIONS> for SigmoidGrad<INPUT_DIMENSIONS> {
    fn input(&self) -> &Tensor<INPUT_DIMENSIONS> {
        &self.input
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for Sigmoid<INPUT_DIMENSIONS>
{
    type Grad = SigmoidGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self::Grad {
        let input = self.input_sig.as_ref().expect("havent forward");
        let input = input.mul_elem(&(1.0 - input)).mul_elem(next_grad);
        Self::Grad { input }
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for Sigmoid<INPUT_DIMENSIONS> {
    type Grad = SigmoidGrad<INPUT_DIMENSIONS>;
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self::Grad) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize> Layer<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for Sigmoid<INPUT_DIMENSIONS>
{
}
