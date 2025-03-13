pub mod layers;
pub mod nn_struct;

use crate::tensor::Tensor;

pub trait Forward<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize> {
    fn forward(&mut self, input: &Tensor<INPUT_DIMENSIONS>) -> Tensor<OUTPUT_DIMENSIONS>;
}

pub trait Backward<const NEXTGRAD_DIMENSIONS: usize> {
    fn backward(&self, next_grad: &Tensor<NEXTGRAD_DIMENSIONS>) -> Self;
}

pub trait Update {
    fn update(self, optimizer: &mut impl Optimizer, grad: Self) -> Self;
}

pub trait Layer<
    const INPUT_DIMENSIONS: usize,
    const OUTPUT_DIMENSIONS: usize,
    const NEXTGRAD_DIMENSIONS: usize,
>: Forward<INPUT_DIMENSIONS, OUTPUT_DIMENSIONS> + Backward<NEXTGRAD_DIMENSIONS>
{
}

pub trait Optimizer {
    fn step<const WEIGHT_DIMENSIONS: usize>(
        &mut self,
        weights: &mut Tensor<WEIGHT_DIMENSIONS>,
        grad: &Tensor<WEIGHT_DIMENSIONS>,
    );
}

pub trait Loss {
    fn loss<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> f32;
    fn loss_grad<const OUTPUT_DIMENSIONS: usize>(
        &mut self,
        prediction: Tensor<OUTPUT_DIMENSIONS>,
        target: Tensor<OUTPUT_DIMENSIONS>,
    ) -> Tensor<OUTPUT_DIMENSIONS>;
}
