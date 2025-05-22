use std::path::Path;

use crate::{
    data::{csv::read_csv, one_hot_encoding::one_hot_encoding},
    nn::{
        layers::{
            conv::{Conv2D, Conv2DGrad},
            flatten::Flatten,
            linear::{Linear, LinearGrad},
            max_pool::MaxPool2D,
            pad::{Pad2D, Pad2DGrad},
            relu::ReLU,
            softmax::Softmax,
        },
        loss::{cross_entropy::CrossEntropy, reduction},
        optimizer::sgd::SGD,
        Backward, Forward, InputGrad, Loss, Optimizer, Update,
    },
    random::{pcg::PCG, RandomGenerator},
    tensor::{
        pad::{pad2d_same_size, PaddingType},
        Tensor,
    },
};

struct Model {
    pad1: Pad2D,
    conv1: Conv2D,
    relu1: ReLU<4>,
    pad2: Pad2D,
    conv2: Conv2D,
    relu2: ReLU<4>,
    max_pool: MaxPool2D,
    flatten: Flatten<4>,
    lin1: Linear,
    relu3: ReLU<2>,
    lin2: Linear,
    softmax: Softmax,
}

impl Model {
    pub fn new() -> Self {
        let mut random_generator = PCG::new(42, 6);
        let mut conv1_random =
            || random_generator.next_normal(0.0, f32::sqrt(2.0 / (1.0 * 3.0 * 3.0)));
        let mut random_generator = PCG::new(42, 6);
        let mut conv2_random =
            || random_generator.next_normal(0.0, f32::sqrt(2.0 / (2.0 * 3.0 * 3.0)));
        let mut random_generator = PCG::new(42, 6);
        let mut lin1_random = || random_generator.next_normal(0.0, f32::sqrt(2.0 / 64.0));
        let mut random_generator = PCG::new(42, 6);
        let mut lin2_random = || random_generator.next_normal(0.0, f32::sqrt(2.0 / 16.0));
        Self {
            pad1: Pad2D::new(pad2d_same_size((8, 8), (3, 3), (1, 1)), PaddingType::Zero),
            conv1: Conv2D::random_init(1, 2, (3, 3), (1, 1), (1, 1), &mut conv1_random),
            relu1: ReLU::default(),
            pad2: Pad2D::new(pad2d_same_size((8, 8), (3, 3), (1, 1)), PaddingType::Zero),
            conv2: Conv2D::random_init(2, 4, (3, 3), (1, 1), (1, 1), &mut conv2_random),
            relu2: ReLU::default(),
            max_pool: MaxPool2D::new((2, 2), (2, 2), (1, 1)),
            flatten: Flatten::new(Some(1), None),
            lin1: Linear::random_init(64, 32, &mut lin1_random),
            relu3: ReLU::default(),
            lin2: Linear::random_init(32, 10, &mut lin2_random),
            softmax: Softmax::default(),
        }
    }
}

impl Forward<4, 2> for Model {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<2> {
        let output = self.pad1.forward(input);
        let output = self.conv1.forward(&output);
        let output = self.relu1.forward(&output);
        let output = self.pad2.forward(&output);
        let output = self.conv2.forward(&output);
        let output = self.relu2.forward(&output);
        let output = self.max_pool.forward(&output);
        let output = self.flatten.forward(&output);
        let output = self.lin1.forward(&output);
        let output = self.relu3.forward(&output);
        let output = self.lin2.forward(&output);
        self.softmax.forward(&output)
    }
}

struct ModelGrad {
    pad1: Pad2DGrad,
    conv1: Conv2DGrad,
    conv2: Conv2DGrad,
    lin1: LinearGrad,
    lin2: LinearGrad,
}

impl InputGrad<4> for ModelGrad {
    fn input(&self) -> Tensor<4> {
        self.pad1.input()
    }
}

impl Backward<4, 2> for Model {
    type Grad = ModelGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        let softmax_grad = self.softmax.backward(next_grad);
        let lin2_grad = self.lin2.backward(&softmax_grad.input());
        let relu3_grad = self.relu3.backward(&lin2_grad.input());
        let lin1_grad = self.lin1.backward(&relu3_grad.input());
        let flatten_grad = self.flatten.backward(&lin1_grad.input());
        let max_pool_grad = self.max_pool.backward(&flatten_grad.input());
        let relu2_grad = self.relu2.backward(&max_pool_grad.input());
        let conv2_grad = self.conv2.backward(&relu2_grad.input());
        let pad2_grad = self.pad2.backward(&conv2_grad.input());
        let relu1_grad = self.relu1.backward(&pad2_grad.input());
        let conv1_grad = self.conv1.backward(&relu1_grad.input());
        let pad1_grad = self.pad1.backward(&conv1_grad.input());
        Self::Grad {
            pad1: pad1_grad,
            conv1: conv1_grad,
            conv2: conv2_grad,
            lin1: lin1_grad,
            lin2: lin2_grad,
        }
    }
}

impl Update for Model {
    type Grad = ModelGrad;
    fn update(&mut self, optimizer: &mut impl Optimizer, grad: &Self::Grad) {
        self.conv1.update(optimizer, &grad.conv1);
        self.conv2.update(optimizer, &grad.conv2);
        self.lin1.update(optimizer, &grad.lin1);
        self.lin2.update(optimizer, &grad.lin2);
    }
}

pub fn get_data(path: &str) -> (Tensor<4>, Tensor<1>) {
    assert!(Path::new(path).is_file(), "File {} does not exist", path);

    let data = read_csv(path, None).expect("should not be error when read csv");
    let &[no_data, _] = data.shape();
    let x = data.submatrix(.., 0..64);
    let x = x.reshape(&[no_data, 8, 8, 1]);
    let y = data.col(65).squeeze(0);
    (x, y)
}

pub fn train() {
    let (data, label) = get_data("data/optdigits.tra");
    let label_one_hot = one_hot_encoding(label, 10);
    let data = &(&data - data.mean()) / data.std();

    let lr = 0.02;
    let epochs = 20;
    let batch_size = 20;

    let no_batch = data.shape()[0] / batch_size;

    let mut model = Model::new();
    let mut loss_fn = CrossEntropy::<reduction::Mean>::default();
    let mut optimizer = SGD::new(lr);

    for e in 0..epochs {
        let mut avg_loss = 0.0;
        for b in 0..no_batch {
            let batch_range = b * batch_size..(b + 1) * batch_size;
            let batch_data = data.subtensor(&[batch_range.clone(), 0..8, 0..8, 0..1]);
            let batch_label_one_hot = label_one_hot.submatrix(batch_range, ..);

            let predict = model.forward(&batch_data);
            let batch_loss = loss_fn.loss(predict.clone(), batch_label_one_hot.clone());

            let loss_grad = loss_fn.loss_grad(predict, batch_label_one_hot);
            let model_grad = model.backward(&loss_grad);

            model.update(&mut optimizer, &model_grad);

            //println!("model_weight: {}", model.lin2.weights().sum());
            avg_loss += batch_loss;
            println!(
                "epoch {} batch ({}/{}): loss {}",
                e, b, no_batch, batch_loss
            );
        }
        avg_loss /= no_batch as f32;
        println!(
            "epoch {}: avg loss {} ---------------------------------------------",
            e, avg_loss
        );
    }

    let (test_data, test_label) = get_data("data/optdigits.tes");
    let test_data = &(&test_data - test_data.mean()) / data.std();
    let predict = model.forward(&test_data);
}
