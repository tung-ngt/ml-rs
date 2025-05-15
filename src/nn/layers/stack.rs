use crate::{
    nn::{optimizer::DynOptimizer, DynLayer, InputGrad},
    tensor::Tensor,
};

use super::DynGrad;

pub struct Stack {
    layers: Vec<Box<dyn DynLayer>>,
}

impl Stack {
    pub fn new(layers: Vec<Box<dyn DynLayer>>) -> Self {
        Self { layers }
    }
}

pub struct StackGrad {
    layers: Vec<DynGrad>,
}

impl InputGrad<2> for StackGrad {
    fn input(&self) -> Tensor<2> {
        self.layers[0].input()
    }
}

impl DynLayer for Stack {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let mut input = input.clone();
        for l in self.layers.iter_mut() {
            input = l.forward(&input);
        }
        input
    }

    fn backward(&self, next_grad: &Tensor<2>) -> DynGrad {
        let mut layers_grad = Vec::with_capacity(self.layers.len());
        let mut next_grad = next_grad.clone();
        for l in self.layers.iter().rev() {
            let grad = l.backward(&next_grad);
            next_grad = grad.input().clone();
            layers_grad.push(grad);
        }

        DynGrad::Stack(StackGrad {
            layers: layers_grad,
        })
    }

    fn update(&mut self, optimizer: &mut DynOptimizer, grad: &DynGrad) {
        let DynGrad::Stack(grad) = grad else {
            return;
        };
        for (l, g) in self.layers.iter_mut().rev().zip(&grad.layers) {
            l.update(optimizer, g);
        }
    }
}
