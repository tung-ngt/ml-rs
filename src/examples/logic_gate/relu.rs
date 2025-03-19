use crate::{
    nn::{
        layers::{linear::Linear, relu::ReLU},
        Backward, Forward, Layer, Optimizer, Update,
    },
    tensor::Tensor,
};

pub struct ReLUModel {
    lin1: Linear,
    relu1: ReLU<2>,
    lin2: Linear,
    relu2: ReLU<2>,
    two_layers: bool,
}

impl ReLUModel {
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
            relu1: ReLU::default(),
            lin2: Linear::new(in_features, out_features),
            relu2: ReLU::default(),
            two_layers,
        }
    }
}

impl Forward<2, 2> for ReLUModel {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let x = self.lin1.forward(input);
        let x = self.relu1.forward(&x);
        if !self.two_layers {
            return x;
        }
        let x = self.lin2.forward(&x);
        self.relu2.forward(&x)
    }
}

impl Backward<2, 2> for ReLUModel {
    fn backward(&self, next_grad: &Tensor<2>) -> Self {
        if self.two_layers {
            let relu2_grad = self.relu2.backward(next_grad);
            let lin2_grad = self.lin2.backward(&relu2_grad.input_grad());
            let relu1_grad = self.relu1.backward(&lin2_grad.input_grad());
            let lin1_grad = self.lin1.backward(&relu1_grad.input_grad());
            Self {
                lin1: lin1_grad,
                relu1: relu1_grad,
                lin2: lin2_grad,
                relu2: relu2_grad,
                two_layers: self.two_layers,
            }
        } else {
            let relu1_grad = self.relu1.backward(next_grad);
            let lin1_grad = self.lin1.backward(&relu1_grad.input_grad());
            Self {
                lin1: lin1_grad,
                relu1: relu1_grad,
                lin2: Linear::new(1, 1),
                relu2: ReLU::default(),
                two_layers: self.two_layers,
            }
        }
    }

    fn input_grad(&self) -> Tensor<2> {
        self.lin1.input_grad()
    }
}

impl Update for ReLUModel {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        if self.two_layers {
            self.lin2 = self.lin2.update(optimizer, grad.lin2);
        }
        self.lin1 = self.lin1.update(optimizer, grad.lin1);
        self
    }
}

impl Layer<2, 2, 2> for ReLUModel {}
