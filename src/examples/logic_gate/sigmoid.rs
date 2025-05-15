use crate::{
    nn::{
        layers::{
            linear::{Linear, LinearGrad},
            sigmoid::Sigmoid,
        },
        Backward, Forward, InputGrad, Optimizer, Update,
    },
    tensor::Tensor,
};

pub struct SigmoidModel {
    pub lin1: Linear,
    pub sigmoid1: Sigmoid<2>,
    pub lin2: Option<Linear>,
    pub sigmoid2: Option<Sigmoid<2>>,
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
            lin2: if two_layers {
                Some(Linear::new(in_features, out_features))
            } else {
                None
            },
            sigmoid2: if two_layers {
                Some(Sigmoid::default())
            } else {
                None
            },
            two_layers,
        }
    }

    pub fn random<G>(
        in_features: usize,
        out_features: usize,
        two_layers: bool,
        random_generator: &mut G,
    ) -> Self
    where
        G: FnMut() -> f32,
    {
        Self {
            lin1: Linear::random_init(
                in_features,
                if two_layers {
                    in_features
                } else {
                    out_features
                },
                random_generator,
            ),
            sigmoid1: Sigmoid::default(),
            lin2: if two_layers {
                Some(Linear::random_init(
                    in_features,
                    out_features,
                    random_generator,
                ))
            } else {
                None
            },
            sigmoid2: if two_layers {
                Some(Sigmoid::default())
            } else {
                None
            },
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
        let x = self.lin2.as_mut().unwrap().forward(&x);
        self.sigmoid2.as_mut().unwrap().forward(&x)
    }
}

pub struct SigmoidModelGrad {
    pub lin1: LinearGrad,
    pub lin2: Option<LinearGrad>,
}

impl InputGrad<2> for SigmoidModelGrad {
    fn input(&self) -> Tensor<2> {
        self.lin1.input().clone()
    }
}

impl Backward<2, 2> for SigmoidModel {
    type Grad = SigmoidModelGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        if self.two_layers {
            let sigmoid2_grad = self.sigmoid2.as_ref().unwrap().backward(next_grad);
            let lin2_grad = self.lin2.as_ref().unwrap().backward(&sigmoid2_grad.input());
            let sigmoid1_grad = self.sigmoid1.backward(&lin2_grad.input());
            let lin1_grad = self.lin1.backward(&sigmoid1_grad.input());
            Self::Grad {
                lin1: lin1_grad,
                lin2: Some(lin2_grad),
            }
        } else {
            let sigmoid1_grad = self.sigmoid1.backward(next_grad);
            let lin1_grad = self.lin1.backward(&sigmoid1_grad.input());
            Self::Grad {
                lin1: lin1_grad,
                lin2: None,
            }
        }
    }
}

impl Update for SigmoidModel {
    type Grad = SigmoidModelGrad;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        if self.two_layers {
            self.lin2
                .as_mut()
                .unwrap()
                .update(optimizer, grad.lin2.as_ref().unwrap());
        }
        self.lin1.update(optimizer, &grad.lin1);
    }
}
