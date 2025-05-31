use crate::{
    nn::{optimizer::DynOptimizer, Backward, DynLayer, Forward, InputGrad, Optimizer, Update},
    tensor::{
        conv::{conv_output_shape, conv_unused_inputs},
        pad::{pad2d_full_size, PaddingSize, PaddingType},
        upsample::dilate_output_shape,
        Tensor,
    },
};

use super::DynGrad;
pub struct Conv2D {
    input_shape: Option<[usize; 4]>,
    input: Option<Tensor<4>>,
    weights: Tensor<4>,
    biases: Tensor<1>,
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl Conv2D {
    pub fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input_shape: None,
            input: None,
            weights: Tensor::filled(&[c_out, kernel_size.0, kernel_size.1, c_in], 1.0),
            biases: Tensor::vector_filled(4, 1.0),
            strides,
            dilations,
        }
    }

    pub fn random_init<G>(
        c_in: usize,
        c_out: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
        random_generator: &mut G,
    ) -> Self
    where
        G: FnMut() -> f32,
    {
        Self {
            input_shape: None,
            input: None,
            weights: Tensor::random(
                &[c_out, kernel_size.0, kernel_size.1, c_in],
                random_generator,
            ),
            biases: Tensor::vector_filled(4, 1.0),
            strides,
            dilations,
        }
    }

    pub fn random_init_with_shape<G>(
        image_shape: &[usize; 4],
        kernel_shape: &[usize; 4],
        strides: (usize, usize),
        dilations: (usize, usize),
        random_generator: &mut G,
    ) -> Self
    where
        G: FnMut() -> f32,
    {
        Self {
            input_shape: Some(*image_shape),
            input: None,
            weights: Tensor::random(kernel_shape, random_generator),
            biases: Tensor::vector_filled(4, 1.0),
            strides,
            dilations,
        }
    }

    pub fn with_shape(
        image_shape: &[usize; 4],
        kernel_shape: &[usize; 4],
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input_shape: Some(*image_shape),
            input: None,
            weights: Tensor::filled(kernel_shape, 1.0),
            biases: Tensor::vector_filled(4, 1.0),
            strides,
            dilations,
        }
    }

    pub fn with_kernel(
        kernels: Tensor<4>,
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input_shape: None,
            input: None,
            weights: kernels,
            biases: Tensor::vector_filled(4, 1.0),
            strides,
            dilations,
        }
    }
}

impl Forward<4, 4> for Conv2D {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        self.input = Some(input.clone());
        input.conv2d(&self.weights.dilate2d(self.dilations), self.strides)
    }
}

pub struct Conv2DGrad {
    input: Tensor<4>,
    weights: Tensor<4>,
    biases: Tensor<1>,
}

impl InputGrad<4> for Conv2DGrad {
    fn input(&self) -> Tensor<4> {
        self.input.clone()
    }
}

impl Backward<4, 4> for Conv2D {
    type Grad = Conv2DGrad;
    fn backward(&self, next_grad: &Tensor<4>) -> Self::Grad {
        let input = self.input.as_ref().expect("havent forward");

        let &[_, h_in, w_in, _] = input.shape();
        let [_, effective_h_k, effective_w_k, _] =
            dilate_output_shape(self.weights.shape(), self.dilations);

        //Dilate the next_grad to accomadate stride > 1
        let next_grad = next_grad.dilate2d(self.strides);

        let input_grad = next_grad
            .pad2d(
                pad2d_full_size((effective_h_k, effective_w_k)),
                PaddingType::Zero,
            )
            .conv2d(
                &self
                    .weights
                    .dilate2d(self.dilations)
                    .transpose(&[3, 1, 2, 0])
                    .reverse(&[1, 2, 3]),
                (1, 1),
            );

        //Handle the case where the kernel and stride does not match up with input image.
        //In which case, some right and bottom inputs are not used. Therefore, we pad the
        //grad with 0 for these inputs.
        let (unused_h, unused_w) =
            conv_unused_inputs((h_in, w_in), (effective_h_k, effective_w_k), self.strides);
        let input_grad = input_grad.pad2d(
            (
                PaddingSize::Diff(0, unused_h),
                PaddingSize::Diff(0, unused_w),
            ),
            PaddingType::Zero,
        );

        //println!("input grad {:?}", input_grad.shape());
        //We also have to do the same thing to the next grad to get the correct weight shape
        let pad_next_grad = next_grad.transpose(&[3, 1, 2, 0]).pad2d(
            (
                PaddingSize::Diff(0, unused_h),
                PaddingSize::Diff(0, unused_w),
            ),
            PaddingType::Zero,
        );

        //println!("pad next grad shape {:?}", pad_next_grad.shape());
        let weight_grad = input
            .transpose(&[3, 1, 2, 0])
            .conv2d(&pad_next_grad, self.dilations)
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
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
    }
}

impl DynLayer for Conv2D {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let input_shape = self
            .input_shape
            .as_ref()
            .expect("must give image shape for dyn conv forward");

        let input = input.reshape(input_shape);
        Forward::forward(self, &input).flatten(Some(1), None)
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        let image_shape = self
            .input_shape
            .as_ref()
            .expect("must give image shape for dyn conv backward");
        let output_shape = conv_output_shape(image_shape, self.weights.shape(), self.strides);
        let next_grad = next_grad.reshape(&output_shape);
        DynGrad::Conv2D(Backward::backward(self, &next_grad))
    }

    fn update(&mut self, optimizer: &mut DynOptimizer, grad: &DynGrad) {
        let DynGrad::Conv2D(grad) = grad else {
            return;
        };

        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
    }
}

#[cfg(test)]
mod conv_layer_tests {
    use crate::nn::layers::conv::{Conv2D, Conv2DGrad};
    use crate::nn::{Backward, Forward};
    use crate::tensor;
    use crate::tensor::pad::{pad2d_same_size, PaddingType};
    use crate::tensor::Tensor;

    #[test]
    fn forward() {
        let a1 = Tensor::<3>::filled(&[5, 5, 1], 1.0);
        let a2 = Tensor::<3>::filled(&[5, 5, 1], 2.0);
        let a = Tensor::stack(&[a1, a2]);
        let kernel = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let pad_a = a.pad2d(pad2d_same_size((5, 5), (3, 3), (1, 1)), PaddingType::Zero);

        let mut conv = Conv2D::with_kernel(kernel, (1, 1), (1, 1));
        let b = conv.forward(&pad_a);

        let c = tensor!(2, 5, 5, 1 => [
            4.0, 6.0, 6.0, 6.0, 4.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            6.0, 9.0, 9.0, 9.0, 6.0,
            4.0, 6.0, 6.0, 6.0, 4.0,

             8.0, 12.0, 12.0, 12.0,  8.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
            12.0, 18.0, 18.0, 18.0, 12.0,
             8.0, 12.0, 12.0, 12.0,  8.0
        ]);

        assert!(b == c);
    }

    #[test]
    fn misalign() {
        let a = Tensor::<4>::filled(&[1, 6, 6, 1], 1.0);
        let kernel = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let mut conv = Conv2D::with_kernel(kernel, (2, 2), (1, 1));
        let b = conv.forward(&a);

        let c = tensor!(1, 2, 2, 1 => [
            9.0, 9.0,
            9.0, 9.0
        ]);

        assert!(b == c, "expected {:?}\ngot {:?}", c.shape(), b.shape());

        let next_grad = tensor!(1, 2, 2, 1 => [
            1.0, 1.0,
            1.0, 1.0
        ]);
        let grad = conv.backward(&next_grad);

        let expected_grad = Conv2DGrad {
            input: tensor!(1, 6, 6, 1 => [
                1.0, 1.0, 2.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 2.0, 1.0, 1.0, 0.0,
                2.0, 2.0, 4.0, 2.0, 2.0, 0.0,
                1.0, 1.0, 2.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 2.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
            weights: tensor!(1, 3, 3, 1 => [
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0
            ]),
            biases: conv.biases.clone(),
        };

        assert!(
            grad.input == expected_grad.input,
            "expected {}\ngot{}",
            expected_grad.input.squeeze(0),
            grad.input.squeeze(0)
        );
        assert!(
            grad.weights == expected_grad.weights,
            "epected {}\ngot {}",
            expected_grad.weights.squeeze(0),
            grad.weights.squeeze(0)
        );
        assert!(grad.biases == expected_grad.biases);
    }

    #[test]
    fn dilate() {
        let a = Tensor::<4>::filled(&[1, 9, 9, 1], 1.0);
        let kernel = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let mut conv = Conv2D::with_kernel(kernel, (1, 1), (3, 3));
        let b = conv.forward(&a);

        let c = tensor!(1, 3, 3, 1 => [
            9.0, 9.0, 9.0,
            9.0, 9.0, 9.0,
            9.0, 9.0, 9.0
        ]);

        assert!(b == c, "expected {:?}\ngot {:?}", c.shape(), b.shape());

        let next_grad = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);
        let grad = conv.backward(&next_grad);

        let expected_grad = Conv2DGrad {
            input: tensor!(1, 9, 9, 1 => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ]),
            weights: tensor!(1, 3, 3, 1 => [
                9.0, 9.0, 9.0,
                9.0, 9.0, 9.0,
                9.0, 9.0, 9.0
            ]),
            biases: conv.biases.clone(),
        };

        assert!(
            grad.input == expected_grad.input,
            "expected {}\ngot{}",
            expected_grad.input.squeeze(0),
            grad.input.squeeze(0)
        );
        assert!(
            grad.weights == expected_grad.weights,
            "epected {}\ngot {}",
            expected_grad.weights.squeeze(0),
            grad.weights.squeeze(0)
        );
        assert!(grad.biases == expected_grad.biases);
    }

    #[test]
    fn stride_and_dilate() {
        let a = Tensor::<4>::filled(&[1, 7, 7, 1], 1.0);
        let kernel = tensor!(1, 3, 3, 1 => [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ]);

        let mut conv = Conv2D::with_kernel(kernel, (2, 2), (2, 2));
        let b = conv.forward(&a);

        let c = tensor!(1, 2, 2, 1 => [
            9.0, 9.0,
            9.0, 9.0
        ]);

        assert!(b == c, "expected {:?}\ngot {:?}", c.shape(), b.shape());

        let next_grad = tensor!(1, 2, 2, 1 => [
            1.0, 1.0,
            1.0, 1.0
        ]);
        let grad = conv.backward(&next_grad);

        let expected_grad = Conv2DGrad {
            input: tensor!(1, 7, 7, 1 => [
                1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 2.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 2.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0
            ]),
            weights: tensor!(1, 3, 3, 1 => [
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0,
                4.0, 4.0, 4.0
            ]),
            biases: conv.biases.clone(),
        };

        assert!(
            grad.input == expected_grad.input,
            "expected {}\ngot{}",
            expected_grad.input.squeeze(0),
            grad.input.squeeze(0)
        );
        assert!(
            grad.weights == expected_grad.weights,
            "epected {}\ngot {}",
            expected_grad.weights.squeeze(0),
            grad.weights.squeeze(0)
        );
        assert!(grad.biases == expected_grad.biases);
    }
}
