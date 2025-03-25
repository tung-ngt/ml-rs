use crate::{
    nn::{Backward, Forward, Layer, Optimizer, Update},
    tensor::Tensor,
};
pub struct PReLU<const INPUT_DIMENSIONS: usize> {
    input_prelu: Tensor<INPUT_DIMENSIONS>,
    negative_slope: Tensor<1>,
}

impl<const INPUT_DIMENSIONS: usize> Default for PReLU<INPUT_DIMENSIONS> {
    fn default() -> Self {
        Self {
            input_prelu: Tensor::new(&[1; INPUT_DIMENSIONS]),
            negative_slope: Tensor::vector_filled(1, 0.01),
        }
    }
}

impl<const INPUT_DIMENSIONS: usize> PReLU<INPUT_DIMENSIONS> {
    pub fn new(negative_slope: f32) -> Self {
        Self {
            input_prelu: Tensor::new(&[1; INPUT_DIMENSIONS]),
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
        self.input_prelu = input.clone();
        input.apply(|x| self.prelu(x))
    }
}

impl<const INPUT_DIMENSIONS: usize> Backward<INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for PReLU<INPUT_DIMENSIONS>
{
    fn backward(&self, next_grad: &Tensor<INPUT_DIMENSIONS>) -> Self {
        Self {
            input_prelu: self
                .input_prelu
                .apply(|x| self.derivative_wrt_input(x))
                .mul_elem(next_grad),
            negative_slope: self.negative_slope.apply(|x| self.derivative_wrt_slope(x)),
        }
    }

    fn input_grad(&self) -> Tensor<INPUT_DIMENSIONS> {
        self.input_prelu.clone()
    }
}

impl<const INPUT_DIMENSIONS: usize> Update for PReLU<INPUT_DIMENSIONS> {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        optimizer.step(&mut self.negative_slope, &grad.negative_slope);
        self
    }
}

impl<const INPUT_DIMENSIONS: usize> Layer<INPUT_DIMENSIONS, INPUT_DIMENSIONS, INPUT_DIMENSIONS>
    for PReLU<INPUT_DIMENSIONS>
{
}
