use crate::{
    nn::{
        layers::conv::Conv2D,
        loss::mse::{reduction, MSE},
        optimizer::{self, sgd::SGD},
        Backward, Forward, Layer, Loss, Optimizer, Update,
    },
    tensor,
    tensor::{conv::PaddingType, utils::pad2d_same_size, Tensor},
};

struct Model {
    conv: Conv2D,
}

impl Model {
    fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            conv: Conv2D::new(c_in, c_out, kernel_size, strides),
        }
    }
}

impl Forward<4, 4> for Model {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<4> {
        self.conv.forward(input)
    }
}

impl Backward<4, 4> for Model {
    fn backward(&self, next_grad: &Tensor<4>) -> Self {
        Self {
            conv: self.conv.backward(next_grad),
        }
    }

    fn input_grad(&self) -> Tensor<4> {
        self.conv.input_grad()
    }
}

impl Update for Model {
    fn update(mut self, optimizer: &mut impl Optimizer, grad: Self) -> Self {
        self.conv = self.conv.update(optimizer, grad.conv);
        self
    }
}

impl Layer<4, 4, 4> for Model {}

pub fn train() {
    let data = tensor!(1, 3, 3, 1 => [
        1.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0

        //1.0, 1.0, 1.0,
        //1.0, 1.0, 1.0,
        //1.0, 1.0, 1.0

        //2.0, 2.0, 2.0,
        //2.0, 2.0, 2.0,
        //2.0, 2.0, 2.0
    ]);

    let label = tensor!(1, 3, 3, 2 => [
        0.25,1.0, 0.3332,2.0, 0.25,1.0,
        0.333,2.0, 0.4444,4.0, 0.33,2.0,
        0.25,1.0, 0.333,2.0, 0.25,1.0

        //1.0, 2.0, 2.0,
        //2.0, 2.0, 2.0,
        //2.0, 2.0, 2.0

        //4.0, 4.0, 4.0,
        //4.0, 4.0, 4.0,
        //4.0, 4.0, 4.0
    ]);

    let epochs = 100;
    let lr = 0.1;

    let mut sgd = SGD::new(lr);
    let mut loss_function = MSE::<reduction::Mean>::default();

    let mut model = Model::new(1, 2, (3, 3), (1, 1));

    let data = data.pad2d(pad2d_same_size((3, 3), (3, 3), (1, 1)), PaddingType::Zero);

    for _e in 0..epochs {
        let predict = model.forward(&data);
        let loss = loss_function.loss(predict.clone(), label.clone());
        println!("loss: {}", loss);
        let loss_grad = loss_function.loss_grad(predict, label.clone());
        let model_grad = model.backward(&loss_grad);

        model = model.update(&mut sgd, model_grad);
    }

    let predict = model.forward(&data);
    println!("{:?}", predict.shape());
    println!(
        "{}",
        predict.subtensor(&[0..1, 0..3, 0..3, 0..2]).squeeze(0)
    );
    //println!(
    //    "{}",
    //    predict.subtensor(&[1..2, 0..3, 0..3, 0..1]).squeeze(0)
    //);
}
