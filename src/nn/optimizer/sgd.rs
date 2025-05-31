use crate::nn::Optimizer;
use crate::tensor::Tensor;

pub struct SGD {
    learning_rate: f32,
    l2_penalty: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            l2_penalty: 1e-7,
        }
    }

    pub fn with_l2_penalty(learning_rate: f32, l2_penalty: f32) -> Self {
        Self {
            learning_rate,
            l2_penalty,
        }
    }
}

impl Optimizer for SGD {
    fn step<const WEIGHT_DIMENSIONS: usize>(
        &mut self,
        weights: &mut Tensor<WEIGHT_DIMENSIONS>,
        grad: &Tensor<WEIGHT_DIMENSIONS>,
    ) {
        *weights = &weights.scale(1.0 - self.l2_penalty) - &(self.learning_rate * grad);
    }
}
