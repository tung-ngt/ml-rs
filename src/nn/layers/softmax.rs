use crate::nn::optimizer::DynOptimizer;
use crate::nn::{Backward, DynLayer, Forward, InputGrad};
use crate::tensor::Tensor;

use super::DynGrad;

#[derive(Default)]
pub struct Softmax {
    input_softmax: Option<Tensor<2>>,
}

impl Forward<2, 2> for Softmax {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let exp = input.apply(|x| x.exp());
        let sum = exp.sum_col();
        let sum = 1.0 / &sum;
        let output = exp.col_mul(&sum);
        self.input_softmax = Some(output.clone());
        output
    }
}

pub struct SoftmaxGrad {
    input: Tensor<2>,
}

impl InputGrad<2> for SoftmaxGrad {
    fn input(&self) -> Tensor<2> {
        self.input.clone()
    }
}

impl Backward<2, 2> for Softmax {
    type Grad = SoftmaxGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        let input_softmax = self.input_softmax.as_ref().expect("havent forward");
        let &[batch, _no_feature] = input_softmax.shape();
        let mut grads: Vec<Tensor<1>> = Vec::new();
        for b in 0..batch {
            let output = input_softmax.row(b);
            let output_t = output.t();
            let diag = output.squeeze(0).to_diag();
            let softmax_jacobian = &diag - &(&output_t * &output);
            let batch_next_grad = next_grad.row(b);
            let softmax_grad = &batch_next_grad * &softmax_jacobian;
            grads.push(softmax_grad.squeeze(0));
        }
        let input_grad = Tensor::stack(&grads);
        Self::Grad { input: input_grad }
    }
}

impl DynLayer for Softmax {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        Forward::forward(self, input)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        DynGrad::Softmax(Backward::backward(self, next_grad))
    }

    fn update(&mut self, _optimizer: &mut DynOptimizer, _grad: &DynGrad) {}
}
