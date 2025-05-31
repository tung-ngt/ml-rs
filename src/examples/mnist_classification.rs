use std::{path::Path, time::Instant};

use crate::{
    data::{csv::read_csv, one_hot_encoding::one_hot_encoding},
    metric::classification::ClassificationReport,
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
        pad::{PaddingSize, PaddingType},
        Tensor,
    },
};

struct Model {
    pad: Pad2D,

    conv1: Conv2D,
    relu1: ReLU<4>,
    max_pool1: MaxPool2D,

    conv2: Conv2D,
    relu2: ReLU<4>,
    max_pool2: MaxPool2D,

    flatten: Flatten<4>,

    lin1: Linear,
    relu3: ReLU<2>,

    lin2: Linear,
    relu4: ReLU<2>,

    lin3: Linear,
    softmax: Softmax,
}

impl Model {
    pub fn new() -> Self {
        let mut random_generator = PCG::new(42, 6);

        let pad = Pad2D::new(
            (PaddingSize::Same(2), PaddingSize::Same(2)),
            PaddingType::Zero,
        );

        let mut conv1_random =
            || random_generator.next_normal(0.0, f32::sqrt(2.0 / (1.0 * 5.0 * 5.0)));
        let conv1 = Conv2D::random_init(1, 6, (5, 5), (1, 1), (1, 1), &mut conv1_random);
        let relu1 = ReLU::default();
        let max_pool1 = MaxPool2D::new((2, 2), (2, 2), (1, 1));

        let mut conv2_random =
            || random_generator.next_normal(0.0, f32::sqrt(2.0 / (6.0 * 5.0 * 5.0)));
        let conv2 = Conv2D::random_init(6, 16, (5, 5), (1, 1), (1, 1), &mut conv2_random);
        let relu2 = ReLU::default();
        let max_pool2 = MaxPool2D::new((2, 2), (2, 2), (1, 1));

        let flatten = Flatten::new(Some(1), None);

        let mut lin1_random = || random_generator.next_normal(0.0, f32::sqrt(2.0 / 400.0));
        let lin1 = Linear::random_init(400, 120, &mut lin1_random);
        let relu3 = ReLU::default();

        let mut lin2_random = || random_generator.next_normal(0.0, f32::sqrt(2.0 / 120.0));
        let lin2 = Linear::random_init(120, 84, &mut lin2_random);
        let relu4 = ReLU::default();

        let mut lin3_random = || random_generator.next_normal(0.0, f32::sqrt(2.0 / 84.0));
        let lin3 = Linear::random_init(84, 10, &mut lin3_random);
        let softmax = Softmax::default();

        Self {
            pad,
            conv1,
            relu1,
            max_pool1,
            conv2,
            relu2,
            max_pool2,
            flatten,
            lin1,
            relu3,
            lin2,
            relu4,
            lin3,
            softmax,
        }
    }
}

impl Forward<4, 2> for Model {
    fn forward(&mut self, input: &Tensor<4>) -> Tensor<2> {
        let output = self.pad.forward(input);

        let output = self.conv1.forward(&output);
        let output = self.relu1.forward(&output);
        let output = self.max_pool1.forward(&output);

        let output = self.conv2.forward(&output);
        let output = self.relu2.forward(&output);
        let output = self.max_pool2.forward(&output);

        let output = self.flatten.forward(&output);

        let output = self.lin1.forward(&output);
        let output = self.relu3.forward(&output);

        let output = self.lin2.forward(&output);
        let output = self.relu4.forward(&output);

        let output = self.lin3.forward(&output);
        self.softmax.forward(&output)
    }
}

struct ModelGrad {
    pad: Pad2DGrad,
    conv1: Conv2DGrad,
    conv2: Conv2DGrad,
    lin1: LinearGrad,
    lin2: LinearGrad,
    lin3: LinearGrad,
}

impl InputGrad<4> for ModelGrad {
    fn input(&self) -> Tensor<4> {
        self.pad.input()
    }
}

impl Backward<4, 2> for Model {
    type Grad = ModelGrad;
    fn backward(&self, next_grad: &Tensor<2>) -> Self::Grad {
        let softmax_grad = self.softmax.backward(next_grad);
        let lin3_grad = self.lin3.backward(&softmax_grad.input());

        let relu4_grad = self.relu4.backward(&lin3_grad.input());
        let lin2_grad = self.lin2.backward(&relu4_grad.input());

        let relu3_grad = self.relu3.backward(&lin2_grad.input());
        let lin1_grad = self.lin1.backward(&relu3_grad.input());

        let flatten_grad = self.flatten.backward(&lin1_grad.input());

        let max_pool2_grad = self.max_pool2.backward(&flatten_grad.input());
        let relu2_grad = self.relu2.backward(&max_pool2_grad.input());
        let conv2_grad = self.conv2.backward(&relu2_grad.input());

        let max_pool1_grad = self.max_pool1.backward(&conv2_grad.input());
        let relu1_grad = self.relu1.backward(&max_pool1_grad.input());
        let conv1_grad = self.conv1.backward(&relu1_grad.input());

        let pad_grad = self.pad.backward(&conv1_grad.input());
        Self::Grad {
            pad: pad_grad,
            conv1: conv1_grad,
            conv2: conv2_grad,
            lin1: lin1_grad,
            lin2: lin2_grad,
            lin3: lin3_grad,
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
        self.lin3.update(optimizer, &grad.lin3);
    }
}

pub fn get_data(path: &str) -> (Tensor<4>, Tensor<1>) {
    assert!(Path::new(path).is_file(), "File {} does not exist", path);

    let data = read_csv(path, Some(1)).expect("should not be error when read csv");
    let &[no_data, _] = data.shape();
    let x = data.submatrix(.., 1..);
    let x = x.reshape(&[no_data, 28, 28, 1]);
    let y = data.col(0).squeeze(1);
    (x, y)
}

pub fn train() {
    let (data, label) = get_data("data/mnist_train.csv");
    let label_one_hot = one_hot_encoding(label.clone(), 10);
    let mean = data.mean();
    let std = data.std();
    let data = &(&data - mean) / std;

    let mut lr = 1.0;
    let min_lr = 0.05;
    let epochs = 10;
    let batch_size = 600;

    let no_sample = data.shape()[0];
    let no_batch = no_sample.div_ceil(batch_size);

    let mut model = Model::new();
    let mut loss_fn = CrossEntropy::<reduction::Mean>::default();
    let mut optimizer = SGD::new(lr);

    let mut random = PCG::new(42, 6);

    for e in 0..epochs {
        let shuffled_index = random.next_shuffle(no_sample);
        let shuffled_data = data.shuffle_batch(&shuffled_index);
        let shuffled_label = label_one_hot.shuffle_batch(&shuffled_index);

        let start = Instant::now();
        let mut avg_loss = 0.0;
        for b in 0..no_batch {
            let batch_range = b * batch_size..usize::min((b + 1) * batch_size, no_sample);
            let batch_data = shuffled_data.subtensor(&[batch_range.clone(), 0..28, 0..28, 0..1]);
            let batch_label_one_hot = shuffled_label.submatrix(batch_range, ..);

            let predict = model.forward(&batch_data);
            let batch_loss = loss_fn.loss(predict.clone(), batch_label_one_hot.clone());

            let loss_grad = loss_fn.loss_grad(predict, batch_label_one_hot);
            let model_grad = model.backward(&loss_grad);

            model.update(&mut optimizer, &model_grad);

            avg_loss += batch_loss;
            println!(
                "epoch {}, lr {}, batch ({}/{}): loss {}",
                e, lr, b, no_batch, batch_loss
            );
        }

        if lr > min_lr {
            lr *= 0.99;
        }
        avg_loss /= no_batch as f32;
        let duration = start.elapsed();
        println!(
            "epoch {} took {:?}: avg loss {} ---------------------------------------------",
            e, duration, avg_loss
        );
    }

    let predict = model.forward(&data);
    let (_predict_prob, predict_label) = predict.max_col();
    let classification_report = ClassificationReport::new(10, predict_label, label);
    println!("train");
    println!(
        "confusion matrix: {}",
        classification_report.confusion_matrix()
    );
    println!("precision: {}", classification_report.precision());
    println!("recall: {}", classification_report.recall());
    println!("f1: {}", classification_report.f1());
    println!("accuracy: {}", classification_report.accuracy());

    let (test_data, test_label) = get_data("data/mnist_test.csv");
    // normalize with the train data value
    let test_data = &(&test_data - mean) / std;
    let predict = model.forward(&test_data);
    let (_predict_prob, predict_label) = predict.max_col();
    let classification_report = ClassificationReport::new(10, predict_label, test_label);
    println!("test");
    println!(
        "confusion matrix: {}",
        classification_report.confusion_matrix()
    );
    println!("precision: {}", classification_report.precision());
    println!("recall: {}", classification_report.recall());
    println!("f1: {}", classification_report.f1());
    println!("accuracy: {}", classification_report.accuracy());
}
