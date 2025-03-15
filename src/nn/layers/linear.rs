use crate::nn::{Backward, Forward, Layer, Optimizer, Update};
use crate::tensor::Tensor;

pub struct Linear {
    input: Tensor<2>,
    pub weights: Tensor<2>,
    pub biases: Tensor<1>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            input: Tensor::new(&[1, in_features]),
            weights: Tensor::filled(&[out_features, in_features], 1.0),
            biases: Tensor::vector_filled(out_features, 1.0),
        }
    }
}

impl Forward<2, 2> for Linear {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let &[out_features, in_features] = self.weights.shape();
        let &[no_inputs, input_features] = input.shape();

        assert!(
            in_features == input_features,
            "Linear cannot dot input {}x{} with weight T {}x{}",
            input_features,
            no_inputs,
            in_features,
            out_features,
        );

        self.input = input.clone();

        (input * &self.weights.t()).row_add(&self.biases)
    }
}

impl Backward<2, 2> for Linear {
    fn backward(&self, next_grad: &Tensor<2>) -> Self {
        Linear {
            input: next_grad * &self.weights,
            weights: &next_grad.t() * &self.input,
            biases: next_grad.sum_row(),
        }
    }

    fn input_grad(&self) -> Tensor<2> {
        self.input.clone()
    }
}

impl Update for Linear {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
        self
    }
}

impl Layer<2, 2, 2> for Linear {}
