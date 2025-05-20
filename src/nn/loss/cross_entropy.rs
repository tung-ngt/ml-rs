use crate::{nn::Loss, tensor::Tensor};

pub struct CrossEntropy;

impl CrossEntropy {
    fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> f32 {
        prediction.apply(|x| -x.ln()).mul_elem(&target).sum()
    }
}
impl Loss<2> for CrossEntropy {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        (1.0 / &prediction).mul_elem(&target)
    }
}
