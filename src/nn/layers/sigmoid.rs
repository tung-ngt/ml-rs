use crate::nn::{Backward, Forward, Layer, Optimizer, Update};
use crate::tensor::Tensor;

pub struct Sigmoid<const INPUT_DIMENSIONS: usize> {
    input_sig: Tensor<INPUT_DIMENSIONS>,
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
        self.input_sig = input.apply(Self::sigmoid);
        self.input_sig.clone()
    }
}

impl<const NEXTGRAD_DIMENSIONS: usize> Backward<NEXTGRAD_DIMENSIONS>
    for Sigmoid<NEXTGRAD_DIMENSIONS>
{
    fn backward(&self, next_grad: &Tensor<NEXTGRAD_DIMENSIONS>) -> Self {
        Self {
            input_sig: next_grad.clone(),
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for Sigmoid<INPUT_DIMENSIONS> {
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize> Layer<INPUT_DIMENSIONS, INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for Sigmoid<INPUT_DIMENSIONS>
{
}
