use crate::{
    nn::{Backward, Forward, Layer, Optimizer, Update},
    tensor::Tensor,
};
pub struct ReLU<const INPUT_DIMENSIONS: usize> {
    input_relu: Tensor<INPUT_DIMENSIONS>,
}

impl<const INPUT_DIMENSIONS: usize> Default for ReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self {
            input_relu: Tensor::new(&[1; INPUT_DIMENSIONS]),
        }
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
        self.input_relu = input.apply(Self::relu);
        self.input_relu.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for ReLU<INPUT_DIMENSIONS>
{
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self {
        Self {
            input_relu: self.input_relu.apply(Self::derivative).mul_elem(next_grad),
        }
    }

    fn input_grad(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input_relu.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for ReLU<INPUT_DIMENSIONS> {
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize> Layer<INPUT_DIMENSIONS, INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for ReLU<INPUT_DIMENSIONS>
{
}
