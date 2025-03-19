use crate::{
    nn::{
        layers::{linear::Linear, sigmoid::Sigmoid},
        Backward, Forward, Layer, Optimizer, Update,
    },
    tensor::Tensor,
};

pub struct SigmoidModel {
    pub lin1: Linear,
    pub sigmoid1: Sigmoid<2>,
    pub lin2: Linear,
    pub sigmoid2: Sigmoid<2>,
    pub two_layers: bool,
}

impl SigmoidModel {
    pub fn new(in_features: usize, out_features: usize, two_layers: bool) -> Self {
        Self {
            lin1: Linear::new(
                in_features,
                if two_layers {
                    in_features
                } else {
                    out_features
                },
            ),
            sigmoid1: Sigmoid::default(),
            lin2: Linear::new(in_features, out_features),
            sigmoid2: Sigmoid::default(),
            two_layers,
        }
    }
}

impl Forward<2, 2> for SigmoidModel {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let x = self.lin1.forward(input);
        let x = self.sigmoid1.forward(&x);
        if !self.two_layers {
            return x;
        }
        let x = self.lin2.forward(&x);
        self.sigmoid2.forward(&x)
    }
}

impl Backward<2, 2> for SigmoidModel {
    fn backward(&self, next_grad: &Tensor<2>) -> Self {
        if self.two_layers {
            let sigmoid2_grad = self.sigmoid2.backward(next_grad);
            let lin2_grad = self.lin2.backward(&sigmoid2_grad.input_grad());
            let sigmoid1_grad = self.sigmoid1.backward(&lin2_grad.input_grad());
            let lin1_grad = self.lin1.backward(&sigmoid1_grad.input_grad());
            Self {
                lin1: lin1_grad,
                sigmoid1: sigmoid1_grad,
                lin2: lin2_grad,
                sigmoid2: sigmoid2_grad,
                two_layers: self.two_layers,
            }
        } else {
            let sigmoid1_grad = self.sigmoid1.backward(next_grad);
            let lin1_grad = self.lin1.backward(&sigmoid1_grad.input_grad());
            Self {
                lin1: lin1_grad,
                sigmoid1: sigmoid1_grad,
                lin2: Linear::new(1, 1),
                sigmoid2: Sigmoid::default(),
                two_layers: self.two_layers,
            }
        }
    }

    fn input_grad(&self) -> Tensor<2> {
        self.lin1.input_grad()
    }
}

impl Update for SigmoidModel {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        if self.two_layers {
            self.lin2 = self.lin2.update(optimizer, grad.lin2);
        }
        self.lin1 = self.lin1.update(optimizer, grad.lin1);
        self
    }
}

impl Layer<2, 2, 2> for SigmoidModel {}
