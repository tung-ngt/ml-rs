use crate::{
    nn::{Backward, Forward, InputGrad, Layer, Optimizer, Update},
    tensor::Tensor,
};

pub struct Flatten<const INPUT_DIMENSIONS: usize> {
    input_shape: Option<[usize; INPUT_DIMENSIONS]>,
    start: usize,
    stop: usize,
}

impl<const INPUT_DIMENSIONS: usize> Flatten<INPUT_DIMENSIONS> {
    pub fn new(start: usize, stop: usize) -> Self {
        Self {
            input_shape: None,
            start,
            stop,
        }
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Forward<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<OUTPUT_DIMENSIONS> {
        self.input_shape = Some(*input.shape());
        input.flatten(&(self.start..self.stop))
    }
}

pub struct FlattenGrad<const INPUT_DIMENSIONS: usize> {
    input: Tensor<INPUT_DIMENSIONS>,
}

impl<const INPUT_DIMENSIONS: usize> InputGrad<INPUT_DIMENSIONS> for FlattenGrad<INPUT_DIMENSIONS> {
    fn input(&self) -> &Tensor<INPUT_DIMENSIONS> {
        &self.input
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Backward<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
    type Grad = FlattenGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<OUTPUT_DIMENSIONS>) -> Self::Grad {
        Self::Grad {
            input: next_grad.reshape(self.input_shape.as_ref().expect("havent forward")),
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for Flatten<INPUT_DIMENSIONS> {
    type Grad = FlattenGrad<INPUT_DIMENSIONS>;
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self::Grad) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>
    Layer<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> for Flatten<INPUT_DIMENSIONS>
{
}
