pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, weights: &mut Matrix, grad: &Matrix) {
        *weights = &*weights - &(self.learning_rate * grad);
    }
}

pub struct MSE;

impl Loss for MSE {
    fn loss(&mut self, prediction: Matrix, target: Matrix) -> f32 {
        assert!(
            prediction.shape() == target.shape(),
            "MSE prediction {:?} and target {:?} have different shape",
            prediction.shape(),
            target.shape()
        );
        (&(&prediction - &target) ^ 2)[(0, 0)]
    }

    fn loss_grad(&mut self, prediction: Matrix, target: Matrix) -> Matrix {
        &(&prediction - &target) * 2.0
    }
}

//pub struct Sigmoid {
//    input_sig: Matrix,
//}
//
//impl Sigmoid {
//    fn sigmoid(x: f32) -> f32 {
//        1.0 / (1.0 + (-x).exp())
//    }
//}
//
//impl Forward for Sigmoid {
//    fn forward(&mut self, input: &Matrix) -> Matrix {
//        self.input_sig = input.apply(Self::sigmoid);
//        self.input_sig.clone()
//    }
//}
//
//impl Backward for Sigmoid {
//    fn backward(&self, next_grad: &Matrix) -> Self {
//        next_grad.t()
//    }
//}
//
//impl Update for Sigmoid {
//    fn update(self, optimizer: &mut impl Optimizer, grad: Self) -> Self {}
//}
//
//impl Layer for Sigmoid {}
