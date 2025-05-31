use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad},
    tensor::{pooling::pooling_output_shape, Tensor},
};

use super::DynGrad;

pub struct MaxPool2D {
    input_shape: Option<[usize; 4]>,
    indicies: Option<Tensor<4>>,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl MaxPool2D {
    pub fn new(
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input_shape: None,
            indicies: None,
            kernel_size,
            strides,
            dilations,
        }
    }

    pub fn with_shape(
        input_shape: &[usize; 4],
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input_shape: Some(*input_shape),
            indicies: None,
            kernel_size,
            strides,
            dilations,
        }
    }
}

impl Forward<4, 4> for MaxPool2D {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        let (x, i) = input.max_pool2d(self.kernel_size, self.strides, self.dilations);
        self.indicies = Some(i);
        self.input_shape = Some(*input.shape());
        x
    }
}

pub struct MaxPool2DGrad {
    input: Tensor<4>,
}

impl InputGrad<4> for MaxPool2DGrad {
    fn input(&self) -> Tensor<4> {
        self.input.clone()
    }
}

impl Backward<4, 4> for MaxPool2D {
    type Grad = MaxPool2DGrad;
    #[allow(non_snake_case)]
    fn backward(&self, next_grad: &Tensor<4>) -> Self::Grad {
        let input_shape = self.input_shape.expect("havent forward");
        let input_strides = Tensor::get_strides(&input_shape);
        let indicies = self.indicies.as_ref().expect("havent forward");
        let mut data = vec![0.0; input_shape.iter().product()];

        let &[B, H_out, W_out, C] = next_grad.shape();

        for b in 0..B {
            for h_out in 0..H_out {
                for w_out in 0..W_out {
                    for c in 0..C {
                        let idx = indicies[&[b, h_out, w_out, c]];
                        let grad = next_grad[&[b, h_out, w_out, c]];

                        let sub_i = (idx as usize) / self.kernel_size.0;
                        let sub_j = (idx as usize) - sub_i * self.kernel_size.0;

                        let i = h_out * self.strides.0 + sub_i * self.dilations.0;
                        let j = w_out * self.strides.1 + sub_j * self.dilations.1;
                        data[b * input_strides[0]
                            + i * input_strides[1]
                            + j * input_strides[2]
                            + c] += grad;
                    }
                }
            }
        }

        Self::Grad {
            input: Tensor::with_data(&input_shape, &input_strides, 0, data.into()),
        }
    }
}

impl DynLayer for MaxPool2D {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let input = input.reshape(
            self.input_shape
                .as_ref()
                .expect("must give image shape for dyn max pool forward"),
        );

        Forward::forward(self, &input).flatten(Some(1), None)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("must give image shape for dyn max pool backward");
        let output_shape =
            pooling_output_shape(input_shape, self.kernel_size, self.strides, self.dilations);
        let next_grad = next_grad.reshape(&output_shape);
        DynGrad::MaxPool2D(Backward::backward(self, &next_grad))
    }

    fn update(&mut self, _optimizer: &mut DynOptimizer, _grad: &DynGrad) {}
}

#[cfg(test)]
mod max_pool_layer {
    use super::MaxPool2D;
    use crate::nn::{Backward, Forward};
    use crate::tensor;

    #[test]
    fn forward() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let b = tensor!(1, 2, 2, 1 => [
            4.0, 5.0,
            6.0, 7.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (2, 2), (1, 1));

        let c = pool.forward(&a);

        assert!(c == b);
    }

    #[test]
    fn backward() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let next_grad = tensor!(1, 2, 2, 1 => [
            1.0, 2.0,
            3.0, 4.0
        ]);

        let expected_input_grad = tensor!(1, 4, 4, 1 => [
            0.0, 0.0, 2.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
            3.0, 0.0, 0.0, 0.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (2, 2), (1, 1));

        let _ = pool.forward(&a);
        let grad = pool.backward(&next_grad);

        assert!(expected_input_grad == grad.input);
    }

    #[test]
    fn misalign() {
        let a = tensor!(1, 4, 5, 1 => [
            1.0, 2.0, 5.0, 3.0, 10.0,
            3.0, 4.0, 2.0, 1.0, 10.0,
            3.0, 2.0, 3.0, 7.0, 10.0,
            6.0, 1.0, 2.0, 1.0, 10.0
        ]);

        let next_grad = tensor!(1, 2, 2, 1 => [
            1.0, 2.0,
            3.0, 4.0
        ]);

        let expected_input_grad = tensor!(1, 4, 5, 1 => [
            0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 4.0, 0.0,
            3.0, 0.0, 0.0, 0.0, 0.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (2, 2), (1, 1));

        let _ = pool.forward(&a);
        let grad = pool.backward(&next_grad);

        assert!(expected_input_grad == grad.input);
    }

    #[test]
    fn forward_overlap() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let b = tensor!(1, 3, 3, 1 => [
            4.0, 5.0, 5.0,
            4.0, 4.0, 7.0,
            6.0, 3.0, 7.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (1, 1), (1, 1));

        let c = pool.forward(&a);

        assert!(c == b);
    }

    #[test]
    fn backward_overlap() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let next_grad = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let expected_input_grad = tensor!(1, 4, 4, 1 => [
            0.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0,
            1.0, 0.0, 0.0, 0.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (1, 1), (1, 1));

        let _ = pool.forward(&a);
        let grad = pool.backward(&next_grad);

        assert!(
            expected_input_grad == grad.input,
            "{}",
            grad.input.subtensor(&[0..1, 0..4, 0..4, 0..1]).squeeze(0)
        );
    }

    #[test]
    fn dilate() {
        let a = tensor!(1, 4, 4, 1 => [
            1.0, 2.0, 5.0, 3.0,
            3.0, 4.0, 2.0, 1.0,
            3.0, 2.0, 3.0, 7.0,
            6.0, 1.0, 2.0, 1.0
        ]);

        let b = tensor!(1, 2, 2, 1 => [
            5.0, 7.0,
            6.0, 4.0
        ]);

        let next_grad = tensor!(1, 2, 2, 1 => [
            1.0, 2.0,
            3.0, 4.0
        ]);

        let expected_input_grad = tensor!(1, 4, 4, 1 => [
            0.0, 0.0, 1.0, 0.0,
            0.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
            3.0, 0.0, 0.0, 0.0
        ]);

        let mut pool = MaxPool2D::new((2, 2), (1, 1), (2, 2));

        let c = pool.forward(&a);
        assert!(b == c, "expected {}\ngot {}", b.squeeze(0), c.squeeze(0));
        let grad = pool.backward(&next_grad);

        assert!(expected_input_grad == grad.input);
    }
}
