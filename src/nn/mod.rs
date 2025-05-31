pub mod dynamic_layer;
pub mod layers;
pub mod loss;
pub mod optimizer;

pub use dynamic_layer::DynLayer;

use crate::tensor::Tensor;

pub trait Forward<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize> {
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<OUTPUT_DIMENSIONS>;
}

pub trait InputGrad<const INPUT_DIMENSIONS: usize> {
    fn input(&self) -> Tensor<INPUT_DIMENSIONS>;
}

pub trait Backward<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize> {
    type Grad: InputGrad<INPUT_DIMENSIONS>;
    fn backward(&self, next_grad: &Tensor<OUTPUT_DIMENSIONS>) -> Self::Grad;
}

pub trait Update {
    type Grad;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad);
}

pub trait Optimizer {
    fn step<const WEIGHT_DIMENSIONS: usize>(
        &mut self,
        weights: &mut Tensor<WEIGHT_DIMENSIONS>,
        grad: &Tensor<WEIGHT_DIMENSIONS>,
    );
}

pub trait Loss<const OUTPUT_DIMENSIONS: usize> {
    fn loss_grad(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS>;
}
