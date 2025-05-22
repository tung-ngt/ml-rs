mod relu;
mod sigmoid;

use relu::ReLUModel;
use sigmoid::SigmoidModel;

use crate::{
    nn::{
        loss::{mse::MSE, reduction},
        optimizer::sgd::SGD,
        Backward, Forward, Loss, Update,
    },
    tensor,
    tensor::Tensor,
};

pub fn and_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            0.0,
            0.0,
            0.0,
            1.0
        ]),
    )
}

pub fn or_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            0.0,
            1.0,
            1.0,
            1.0
        ]),
    )
}

pub fn xor_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            0.0,
            1.0,
            1.0,
            0.0
        ]),
    )
}

pub fn nand_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            1.0,
            1.0,
            1.0,
            0.0
        ]),
    )
}
pub fn nor_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            1.0,
            0.0,
            0.0,
            0.0
        ]),
    )
}

pub fn nxor_dataset() -> (Tensor<2>, Tensor<2>) {
    (
        tensor!(4, 2 => [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        tensor!(4, 1 => [
            1.0,
            0.0,
            0.0,
            1.0
        ]),
    )
}

pub fn train_sigmoid<G>(
    dataset: (Tensor<2>, Tensor<2>),
    two_layers: bool,
    epochs: usize,
    lr: f32,
    random_generator: Option<&mut G>,
) where
    G: FnMut() -> f32,
{
    let (datas, labels) = dataset;

    let mut model = if let Some(random_generator) = random_generator {
        SigmoidModel::random(2, 1, two_layers, random_generator)
    } else {
        SigmoidModel::new(2, 1, two_layers)
    };

    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);
    println!("start lin1_weights = {:.10}", model.lin1.weights());
    println!("start lin1_biases = {:.10}", model.lin1.biases());

    if two_layers {
        println!(
            "start lin2_weights = {:.10}",
            model.lin2.as_ref().unwrap().weights()
        );
        println!(
            "start lin2_biases = {:.10}",
            model.lin2.as_ref().unwrap().biases()
        );
    }
    println!("---------------------------------------------------------------");

    let mut optimizer = SGD::new(lr);
    let mut loss_function = MSE::<reduction::Mean>::default();

    for i in 0..epochs {
        let output = model.forward(&datas);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let model_grad = model.backward(&loss_grad);
        println!("({}) loss = {:.10}", i, loss);

        model.update(&mut optimizer, &model_grad);
    }
    let test = datas.clone();
    let output = model.forward(&test);

    println!("---------------------------------------------------------------");
    println!("output = {:.4}", output);
    println!("end lin1_weights = {:.10}", model.lin1.weights());
    println!("end lin1_biases = {:.10}", model.lin1.biases());

    if two_layers {
        println!(
            "end lin2_weights = {:.10}",
            model.lin2.as_ref().unwrap().weights()
        );
        println!(
            "end lin2_biases = {:.10}",
            model.lin2.as_ref().unwrap().biases()
        );
    }
}

pub fn train_relu<G>(
    dataset: (Tensor<2>, Tensor<2>),
    two_layers: bool,
    epochs: usize,
    lr: f32,
    random_generator: Option<&mut G>,
) where
    G: FnMut() -> f32,
{
    let (datas, labels) = dataset;

    let mut model = if let Some(random_generator) = random_generator {
        ReLUModel::random(2, 1, two_layers, random_generator)
    } else {
        ReLUModel::new(2, 1, two_layers)
    };

    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);
    println!("start lin1_weights = {:.10}", model.lin1.weights());
    println!("start lin1_biases = {:.10}", model.lin1.biases());

    if two_layers {
        println!(
            "start lin2_weights = {:.10}",
            model.lin2.as_ref().unwrap().weights()
        );
        println!(
            "start lin2_biases = {:.10}",
            model.lin2.as_ref().unwrap().biases()
        );
    }
    println!("---------------------------------------------------------------");

    let mut optimizer = SGD::new(lr);
    let mut loss_function = MSE::<reduction::Mean>::default();

    for i in 0..epochs {
        let output = model.forward(&datas);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let model_grad = model.backward(&loss_grad);
        println!("({}) loss = {:.10}", i, loss);
        //println!("({}) grad_weights = {:.10}", i, grad.weights);
        //println!("({}) grad_biases = {:.10}", i, grad.biases);
        //println!("({}) weights = {:.10}", i, lin.weights);
        //println!("({}) biases = {:.10}", i, lin.biases);
        model.update(&mut optimizer, &model_grad);
    }
    let test = datas.clone();
    let output = model.forward(&test);

    println!("---------------------------------------------------------------");
    println!("output = {:.4}", output);
    println!("end lin1_weights = {:.10}", model.lin1.weights());
    println!("end lin1_biases = {:.10}", model.lin1.biases());

    if two_layers {
        println!(
            "end lin2_weights = {:.10}",
            model.lin2.as_ref().unwrap().weights()
        );
        println!(
            "end lin2_biases = {:.10}",
            model.lin2.as_ref().unwrap().biases()
        );
    }
}
