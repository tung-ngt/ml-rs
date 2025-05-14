use crate::{
    nn::{Backward, Forward, Layer, Optimizer, Update},
    tensor::{conv::PaddingType, utils::pad2d_full_size, Tensor},
};
pub struct Conv2D {
    input: Tensor<4>,
    weights: Tensor<4>,
    biases: Tensor<1>,
    strides: (usize, usize),
}

impl Forward<4, 4> for Conv2D {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        self.input = input.clone();
        input.conv2d(&self.weights, self.strides)
    }
}

impl Backward<4, 4> for Conv2D {
    fn backward(&self, next_grad: &Tensor<4>) -> Self {
        let &[_b, h_out, w_out, _c_out] = next_grad.shape();
        let &[_, h_k, w_k, _c_in] = self.weights.shape();
        let input_grad = next_grad
            .pad2d(
                pad2d_full_size((h_out, w_out), (h_k, w_k), self.strides),
                PaddingType::Zero,
            )
            .conv2d(
                &self.weights.transpose(&[3, 0, 1, 2]).reverse(&[1, 2, 3]),
                self.strides,
            );

        let weight_grad = self.input.conv2d(next_grad, self.strides);

        Self {
            input: input_grad,
            strides: self.strides,
            weights: weight_grad,
            biases: self.biases.clone(),
        }
    }
    fn input_grad(&self) -> Tensor<4> {
        self.input.clone()
    }
}

impl Update for Conv2D {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
        self
    }
}

//impl Layer<4, 4, 4> for Conv2D {}
