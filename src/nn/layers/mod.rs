pub mod conv;
pub mod flatten;
pub mod leaky_relu;
pub mod linear;
pub mod max_pool;
pub mod pad;
pub mod prelu;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod stack;

use conv::Conv2DGrad;
use flatten::FlattenGrad;
use leaky_relu::LeakyReLUGrad;
use linear::LinearGrad;
use max_pool::MaxPool2DGrad;
use pad::Pad2DGrad;
use prelu::PReLUGrad;
use relu::ReLUGrad;
use sigmoid::SigmoidGrad;
use softmax::SoftmaxGrad;
use stack::StackGrad;

use crate::tensor::Tensor;

use super::InputGrad;

pub enum DynGrad {
    Conv2D(Conv2DGrad),
    Linear(LinearGrad),
    Flatten(FlattenGrad<2>),
    LeakyReLU(LeakyReLUGrad<2>),
    PReLU(PReLUGrad<2>),
    ReLU(ReLUGrad<2>),
    Sigmoid(SigmoidGrad<2>),
    Stack(StackGrad),
    MaxPool2D(MaxPool2DGrad),
    Pad2D(Pad2DGrad),
    Softmax(SoftmaxGrad),
}

impl DynGrad {
    pub fn input(&self) -> Tensor<2> {
        match self {
            Self::Conv2D(g) => g.input().flatten(Some(1), None),
            Self::MaxPool2D(g) => g.input().flatten(Some(1), None),
            Self::Pad2D(g) => g.input().flatten(Some(1), None),
            Self::Linear(g) => g.input().clone(),
            Self::Flatten(g) => g.input().clone(),
            Self::LeakyReLU(g) => g.input().clone(),
            Self::PReLU(g) => g.input().clone(),
            Self::ReLU(g) => g.input().clone(),
            Self::Sigmoid(g) => g.input().clone(),
            Self::Stack(g) => g.input(),
            Self::Softmax(g) => g.input(),
        }
    }
}
