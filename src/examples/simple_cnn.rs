use crate::{
    nn::{
        layers::{
            conv::{Conv2D, Conv2DGrad},
            flatten::Flatten,
        },
        loss::{mse::MSE, reduction},
        optimizer::sgd::SGD,
        Backward, Forward, InputGrad, Loss, Optimizer, Update,
    },
    tensor,
    tensor::{
        pad::{pad2d_same_size, PaddingType},
        Tensor,
    },
};

struct Model {
    conv: Conv2D,
    flatten: Flatten<4>,
}

impl Model {
    fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            conv: Conv2D::new(c_in, c_out, kernel_size, strides, (1, 1)),
            flatten: Flatten::new(Some(1), None),
        }
    }
}

impl Forward<4, 2> for Model {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<2> {
        let input = self.conv.forward(input);
        self.flatten.forward(&input)
    }
}

struct ModelGrad {
    conv: Conv2DGrad,
}

impl InputGrad<4> for ModelGrad {
    fn input(&self) -> Tensor<4> {
        self.conv.input().clone()
    }
}

impl Backward<4, 2> for Model {
    type Grad = ModelGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        let flatten_grad = self.flatten.backward(next_grad);
        let conv_grad = self.conv.backward(&flatten_grad.input());
        Self::Grad { conv: conv_grad }
    }
}

impl Update for Model {
    type Grad = ModelGrad;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        self.conv.update(optimizer, &grad.conv);
    }
}

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

    let label = tensor!(1, 18 => [
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

    let epochs = 500;
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

        model.update(&mut sgd, &model_grad);
    }

    let predict = model.forward(&data);
    println!("{:.5}", predict);
}
