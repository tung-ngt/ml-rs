use crate::linear::Matrix;

pub trait Forward {
    fn forward(&mut self, input: &Matrix) -> Matrix;
}

pub trait Backward {
    fn backward(&self, next_grad: &Matrix) -> Self;
}

pub trait Update {
    fn update(self, optimizer: &mut impl Optimizer, grad: Self) -> Self;
}

pub trait Layer: Forward + Backward {}

pub struct Linear {
    input: Matrix,
    pub weights: Matrix,
    pub biases: Matrix,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            input: Matrix::new(1, in_features),
            weights: Matrix::filled(out_features, in_features, 1.0),
            biases: Matrix::filled(out_features, 1, 1.0),
        }
    }
}

impl Forward for Linear {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let (out_features, in_features) = self.weights.shape();
        let (no_inputs, input_features) = input.shape();

        assert!(
            in_features == input_features,
            "Linear cannot dot input {}x{} with weight T {}x{}",
            input_features,
            no_inputs,
            in_features,
            out_features,
        );

        self.input = input.clone();
        &(input * &self.weights.t()) + &self.biases.t()
    }
}

impl Backward for Linear {
    fn backward(&self, next_grad: &Matrix) -> Self {
        let (_, in_features) = self.weights.shape();
        Linear {
            input: Matrix::new(in_features, 1),
            weights: &next_grad.t() * &self.input,
            biases: next_grad.t(),
        }
    }
}

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

pub trait Optimizer {
    fn step(&mut self, weights: &mut Matrix, grad: &Matrix);
}

impl Update for Linear {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        optimizer.step(&mut self.weights, &grad.weights);
        optimizer.step(&mut self.biases, &grad.biases);
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, weights: &mut Matrix, grad: &Matrix) {
        *weights = &*weights - &(self.learning_rate * grad);
    }
}

pub struct MSE;

pub trait Loss {
    fn loss(&mut self, prediction: Matrix, target: Matrix) -> f32;
    fn loss_grad(&mut self, prediction: Matrix, target: Matrix) -> Matrix;
}

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

impl Layer for Linear {}

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
