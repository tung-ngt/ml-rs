use crate::{
    nn::{Backward, Forward, InputGrad, Layer, Optimizer, Update},
    tensor::{conv::PaddingType, utils::pad2d_full_size, Tensor},
};
pub struct Conv2D {
    input: Option<Tensor<4>>,
    weights: Tensor<4>,
    biases: Tensor<1>,
    strides: (usize, usize),
}

impl Conv2D {
    pub fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            input: None,
            weights: Tensor::filled(&[c_out, kernel_size.0, kernel_size.1, c_in], 1.0),
            biases: Tensor::vector_filled(4, 1.0),
            strides,
        }
    }

    //pub fn random_init<G>(in_features: usize, out_features: usize, random_generator: &mut G) -> Self
    //where
    //    G: FnMut() -> f32,
    //{
    //    Self {
    //        input: Tensor::new(&[1, in_features]),
    //        weights: Tensor::matrix_random(out_features, in_features, random_generator),
    //        biases: Tensor::vector_random(out_features, random_generator),
    //    }
    //}
}

impl Forward<4, 4> for Conv2D {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        self.input = Some(input.clone());
        input.conv2d(&self.weights, self.strides)
    }
}

pub struct Conv2DGrad {
    input: Tensor<4>,
    weights: Tensor<4>,
    biases: Tensor<1>,
}

impl InputGrad<4> for Conv2DGrad {
    fn input(&self) -> &Tensor<4> {
        &self.input
    }
}

impl Backward<4, 4> for Conv2D {
    type Grad = Conv2DGrad;
    fn backward(&self, next_grad: &Tensor<4>) -> Self::Grad {
        let &[_b, h_out, w_out, _c_out] = next_grad.shape();
        let &[_, h_k, w_k, _c_in] = self.weights.shape();
        let input_grad = next_grad
            .pad2d(
                pad2d_full_size((h_out, w_out), (h_k, w_k), self.strides),
                PaddingType::Zero,
            )
            .conv2d(
                &self.weights.transpose(&[3, 1, 2, 0]).reverse(&[1, 2, 3]),
                self.strides,
            );

        let weight_grad = self
            .input
            .as_ref()
            .expect("havent forward")
            .transpose(&[3, 1, 2, 0])
            .conv2d(&next_grad.transpose(&[3, 1, 2, 0]), self.strides)
            .transpose(&[3, 1, 2, 0]);

        Self::Grad {
            input: input_grad,
            weights: weight_grad,
            biases: self.biases.clone(),
        }
    }
}

impl Update for Conv2D {
    type Grad = Conv2DGrad;
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self::Grad) -> Self {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
        self
    }
}

impl Layer<4, 4> for Conv2D {}
