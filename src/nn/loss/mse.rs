use std::marker;

use super::reduction;
use crate::nn::Loss;
use crate::tensor::Tensor;

#[derive(Default)]
pub struct MSE<T> {
    _marker: marker::PhantomData<T>,
}

impl<T: reduction::Reduction> MSE<T> {
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
        &(&prediction - &target) ^ 2
    }

    fn internal_loss_grad<const OUTPUT_DIMENSIONS: usize>(
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        &(&prediction - &target) * 2.0
    }
}

impl MSE<reduction::Mean> {
    pub fn loss<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> f32 {
        let loss_tensor = Self::internal_loss(prediction, target);
        loss_tensor.sum() / (loss_tensor.no_elements() as f32)
    }
}

impl<const OUTPUT_DIMENSIONS: usize> Loss<OUTPUT_DIMENSIONS> for MSE<reduction::Mean> {
    fn loss_grad(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let grad = Self::internal_loss_grad(prediction, target);
        &grad / (grad.no_elements() as f32)
    }
}

impl MSE<reduction::Sum> {
    pub fn loss<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> f32 {
        let loss_tensor = Self::internal_loss(prediction, target);
        loss_tensor.sum()
    }
}

impl<const OUTPUT_DIMENSIONS: usize> Loss<OUTPUT_DIMENSIONS> for MSE<reduction::Sum> {
    fn loss_grad(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        Self::internal_loss_grad(prediction, target)
    }
}

impl MSE<reduction::NoReduction> {
    pub fn loss<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        Self::internal_loss(prediction, target)
    }
}

impl<const OUTPUT_DIMENSIONS: usize> Loss<OUTPUT_DIMENSIONS> for MSE<reduction::NoReduction> {
    fn loss_grad(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        Self::internal_loss_grad(prediction, target)
    }
}

impl MSE<reduction::MeanBatch> {
    pub fn loss<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let loss_tensor = Self::internal_loss(prediction, target);
        let batch = loss_tensor.shape()[0];
        &loss_tensor.sum_dim(0) / (batch as f32)
    }
}

impl<const OUTPUT_DIMENSIONS: usize> Loss<OUTPUT_DIMENSIONS> for MSE<reduction::MeanBatch> {
    fn loss_grad(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS> {
        let grad = Self::internal_loss_grad(prediction, target);
        let batch = grad.shape()[0];
        &grad / (batch as f32)
    }
}
