use crate::nn::{Backward, Forward, InputGrad};
use crate::tensor::Tensor;

pub struct Softmax {
    input_sig: Option<Tensor<2>>,
}

impl Default for Softmax {
    fn default() -> Self {
        Self { input_sig: None }
    }
}

impl Forward<2, 2> for Softmax {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let exp = input.apply(|x| x.exp());
        let sum = 1.0 / &exp.sum_row();

        let output = exp.col_mul(&sum);
        self.input_sig = Some(output.clone());
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
        let input = self.input_sig.as_ref().expect("havent forward");
        let input = input.mul_elem(&(1.0 - input)).mul_elem(next_grad);
        Self::Grad { input }
    }
}
