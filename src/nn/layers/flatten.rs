use crate::{
    nn::{Backward, Forward, Layer, Optimizer, Update},
    tensor::Tensor,
};

pub struct Flatten<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
    input_shape: [usize; INPUT_DIMENSIONS],
    start: usize,
    stop: usize,
}

impl<const INPUT_DIMENSIONS: usize> Flatten<INPUT_DIMENSIONS> {
    pub fn new(start: usize, stop: usize) -> Self {
        Self {
            input: Tensor::empty(),
            input_shape: [0; INPUT_DIMENSIONS],
            start,
            stop,
        }
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Forward<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<OUTPUT_DIMENSIONS> {
        self.input_shape = *input.shape();
        input.flatten(&(self.start..self.stop))
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Backward<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
    fn backward(&self, next_grad: &Tensor<OUTPUT_DIMENSIONS>) -> Self {
        Self {
            input: next_grad.reshape(&self.input_shape),
            input_shape: self.input_shape,
            stop: self.stop,
            start: self.start,
        }
    }

    fn input_grad(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for Flatten<INPUT_DIMENSIONS> {
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Layer<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
}
