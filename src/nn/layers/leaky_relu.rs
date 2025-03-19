use crate::{
    nn::{Backward, Forward, Layer, Optimizer, Update},
    tensor::Tensor,
};
pub struct LeakyReLU<const INPUT_DIMENSIONS: usize> {
    input_leaky_relu: Tensor<INPUT_DIMENSIONS>,
    negative_slope: f32,
}

impl<const INPUT_DIMENSIONS: usize> Default for LeakyReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self {
            input_leaky_relu: Tensor::new(&[1; INPUT_DIMENSIONS]),
            negative_slope: 0.01,
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> LeakyReLU<INPUT_DIMENSIONS> {
    pub fn new(negative_slope: f32) -> Self {
        Self {
            input_leaky_relu: Tensor::new(&[1; INPUT_DIMENSIONS]),
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
        self.input_leaky_relu = input.clone();
        input.apply(|x| self.leaky_relu(x))
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for LeakyReLU<INPUT_DIMENSIONS>
{
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self {
        Self {
            input_leaky_relu: self
                .input_leaky_relu
                .apply(|x| self.derivative(x))
                .mul_elem(next_grad),
            negative_slope: self.negative_slope,
        }
    }

    fn input_grad(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input_leaky_relu.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for LeakyReLU<INPUT_DIMENSIONS> {
    fn update(self, _optimizer: &mut impl Optimizer, _grad: Self) -> Self {
        self
    }
}

impl<const INPUT_DIMENSIONS: usize> Layer<INPUT_DIMENSIONS, INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for LeakyReLU<INPUT_DIMENSIONS>
{
}
