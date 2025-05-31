use std::marker;

use crate::{nn::Loss, tensor::Tensor};

use super::reduction;

#[derive(Default)]
pub struct CrossEntropy<T> {
    _marker: marker::PhantomData<T>,
}

impl<T: reduction::Reduction> CrossEntropy<T> {
    fn internal_loss<const OUTPUT_DIMENSIONS: usize>(
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        assert!(
            prediction.shape() == target.shape(),
            "MSE prediction {:?} and target {:?} have different shape",
            prediction.shape(),
            target.shape()
        );
        prediction.apply(|x| -x.max(1e-7).ln()).mul_elem(&target)
    }

    fn internal_loss_grad<const OUTPUT_DIMENSIONS: usize>(
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        (-1.0 / &prediction.apply(|x| x.max(1e-7))).mul_elem(&target)
    }
}

impl CrossEntropy<reduction::Mean> {
    pub fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> f32 {
        let loss_tensor = Self::internal_loss(prediction, target);
        loss_tensor.sum() / (loss_tensor.no_elements() as f32)
    }
}

impl Loss<2> for CrossEntropy<reduction::Mean> {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        let grad = Self::internal_loss_grad(prediction, target);
        &grad / (grad.no_elements() as f32)
    }
}

impl CrossEntropy<reduction::Sum> {
    pub fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> f32 {
        let loss_tensor = Self::internal_loss(prediction, target);
        loss_tensor.sum()
    }
}

impl Loss<2> for CrossEntropy<reduction::Sum> {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        Self::internal_loss_grad(prediction, target)
    }
}

impl CrossEntropy<reduction::NoReduction> {
    pub fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        Self::internal_loss(prediction, target)
    }
}

impl Loss<2> for CrossEntropy<reduction::NoReduction> {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        Self::internal_loss_grad(prediction, target)
    }
}

impl CrossEntropy<reduction::MeanBatch> {
    pub fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<1> {
        let loss_tensor = Self::internal_loss(prediction, target);
        let batch = loss_tensor.shape()[0] as f32;
        let loss_tensor = loss_tensor.sum_row();
        &loss_tensor / batch
    }
}

impl Loss<2> for CrossEntropy<reduction::MeanBatch> {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        let grad = Self::internal_loss_grad(prediction, target);
        let batch = grad.shape()[0] as f32;
        &grad / batch
    }
}

impl CrossEntropy<reduction::MeanFeature> {
    pub fn loss(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<1> {
        let loss_tensor = Self::internal_loss(prediction, target);
        let no_features = loss_tensor.shape()[1] as f32;
        let loss_tensor = loss_tensor.sum_col();
        &loss_tensor / no_features
    }
}

impl Loss<2> for CrossEntropy<reduction::MeanFeature> {
    fn loss_grad(&mut self, prediction: Tensor<2>, target: Tensor<2>) -> Tensor<2> {
        let grad = Self::internal_loss_grad(prediction, target);
        let no_features = grad.shape()[1] as f32;
        &grad / no_features
    }
}
