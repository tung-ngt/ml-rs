use sgd::SGD;

use crate::tensor::Tensor;

use super::Optimizer;

pub mod sgd;

pub enum DynOptimizer {
    SGD(SGD),
}

impl DynOptimizer {
    pub fn step<const WEIGHT_DIMENSIONS: usize>(
        &mut self,
        weights: &mut Tensor<WEIGHT_DIMENSIONS>,
        grad: &Tensor<WEIGHT_DIMENSIONS>,
    ) {
        match self {
            Self::SGD(o) => o.step(weights, grad),
        }
    }
}
