use crate::tensor::Tensor;

use super::{layers::DynGrad, optimizer::DynOptimizer};

pub trait DynLayer {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2>;
    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad;
    fn update(&mut self, optimizer: &mut DynOptimizer, grad: &DynGrad);
}
