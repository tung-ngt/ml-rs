use crate::{
    nn::{
        layers::{linear::Linear, sigmoid::Sigmoid},
        loss::mse::{reduction, MSE},
        optimizer::sgd::SGD,
        Backward, Forward, Layer, Loss, Optimizer, Update,
    },
    tensor,
    tensor::Tensor,
};

struct M {
    lin: Linear,
    sigmoid: Sigmoid<2>,
}

impl M {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            lin: Linear::new(in_features, out_features),
            sigmoid: Sigmoid::default(),
        }
    }
}

impl Forward<2, 2> for M {
    fn forward(&mut self, input: &Tensor<2>) -> Tensor<2> {
        let x = self.lin.forward(input);
        self.sigmoid.forward(&x)
    }
}

impl Backward<2, 2> for M {
    fn backward(&self, next_grad: &Tensor<2>) -> Self {
        let sigmoid_grad = self.sigmoid.backward(next_grad);
        let lin_grad = self.lin.backward(&sigmoid_grad.input_grad());
        Self {
            lin: lin_grad,
            sigmoid: sigmoid_grad,
        }
    }

    fn input_grad(&self) -> Tensor<2> {
        self.lin.input_grad()
    }
}

impl Update for M {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        self.lin = self.lin.update(optimizer, grad.lin);
        self
    }
}

impl Layer<2, 2, 2> for M {}

pub fn train() {
    let datas = tensor!(4, 2 => [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]);

    //let labels = tensor!(5, 1 => [
    //    10.0,
    //    8.0,
    //    4.0,
    //    5.0,
    //    -2.0
    //]);

    let labels = tensor!(4, 1 => [
        0.0,
        0.0,
        0.0,
        1.0
    ]);

    //let &[no_data, _] = labels.shape();

    let mut model = M::new(2, 1);
    let mut optimizer = SGD::new(0.1);
    let mut loss_function = MSE::<reduction::Mean>::default();

    const EPOCHS: usize = 1000;
    for i in 0..EPOCHS {
        let output = model.forward(&datas);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let model_grad = model.backward(&loss_grad);
        println!("({}) loss = {:.10}", i, loss);
        //println!("({}) grad_weights = {:.10}", i, grad.weights);
        //println!("({}) grad_biases = {:.10}", i, grad.biases);
        //println!("({}) weights = {:.10}", i, lin.weights);
        //println!("({}) biases = {:.10}", i, lin.biases);
        model = model.update(&mut optimizer, model_grad);
    }
    //let test = tensor!(1, 2 => [2.0, -5.0]);
    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);
}
