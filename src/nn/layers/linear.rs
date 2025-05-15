use crate::nn::optimizer::DynOptimizer;
use crate::nn::{Backward, DynLayer, Forward, InputGrad, Optimizer, Update};
use crate::tensor::Tensor;

use super::DynGrad;

pub struct Linear {
    input: Option<Tensor<2>>,
    weights: Tensor<2>,
    biases: Tensor<1>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            input: None,
            weights: Tensor::filled(&[out_features, in_features], 1.0),
            biases: Tensor::vector_filled(out_features, 1.0),
        }
    }

    pub fn random_init<G>(in_features: usize, out_features: usize, random_generator: &mut G) -> Self
    where
        G: FnMut() -> f32,
    {
        Self {
            input: None,
            weights: Tensor::matrix_random(out_features, in_features, random_generator),
            biases: Tensor::vector_random(out_features, random_generator),
        }
    }

    pub fn weights(&self) -> &Tensor<2> {
        &self.weights
    }

    pub fn biases(&self) -> &Tensor<1> {
        &self.biases
    }
}

impl Forward<2, 2> for Linear {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let &[out_features, in_features] = self.weights.shape();
        let &[no_inputs, input_features] = input.shape();

        assert!(
            in_features == input_features,
            "Linear cannot dot input {}x{} with weight T {}x{}",
            no_inputs,
            input_features,
            in_features,
            out_features,
        );

        self.input = Some(input.clone());

        (input * &self.weights.t()).row_add(&self.biases)
    }
}

pub struct LinearGrad {
    input: Tensor<2>,
    weights: Tensor<2>,
    biases: Tensor<1>,
}

impl InputGrad<2> for LinearGrad {
    fn input(&self) -> Tensor<2> {
        self.input.clone()
    }
}

impl Backward<2, 2> for Linear {
    type Grad = LinearGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        Self::Grad {
            input: next_grad * &self.weights,
            weights: &next_grad.t() * self.input.as_ref().expect("havent forward"),
            biases: next_grad.sum_row(),
        }
    }
}

impl Update for Linear {
    type Grad = LinearGrad;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
    }
}

impl DynLayer for Linear {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        Forward::forward(self, input)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        DynGrad::Linear(Backward::backward(self, next_grad))
    }

    fn update(&mut self, optimizer: &mut DynOptimizer, grad: &DynGrad) {
        let DynGrad::Linear(grad) = grad else {
            return;
        };
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
    }
}
