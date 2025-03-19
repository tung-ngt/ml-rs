mod relu;
mod sigmoid;

use relu::ReLUModel;
use sigmoid::SigmoidModel;

use crate::{
    nn::{
        loss::mse::{reduction, MSE},
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

pub fn train_sigmoid(dataset: (Tensor<2>, Tensor<2>), two_layers: bool, epochs: usize, lr: f32) {
    let (datas, labels) = dataset;

    let mut model = SigmoidModel::new(2, 1, two_layers);

    //model.lin1.weights = tensor!(2, 2 => [
    //    -6.3977, -6.3822,
    //    -4.7966, -4.7935
    //]);
    //
    //model.lin1.biases = tensor!(2 => [
    //    2.6063, 7.1380
    //]);
    //
    //model.lin2.weights = tensor!(1, 2 => [
    //    -9.9069,  9.7319
    //]);
    //
    //model.lin2.biases = tensor!(1 => [
    //    -4.6067
    //]);

    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);

    let mut optimizer = SGD::new(lr);
    let mut loss_function = MSE::<reduction::Mean>::default();

    for i in 0..epochs {
        let output = model.forward(&datas);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let model_grad = model.backward(&loss_grad);
        println!("({}) loss = {:.10}", i, loss);

        println!(
            "({}) grad_lin1_weights = {:.10}",
            i, model_grad.lin1.weights
        );
        println!("({}) grad_lin1_biases = {:.10}", i, model_grad.lin1.biases);

        println!(
            "({}) grad_lin2_weights = {:.10}",
            i, model_grad.lin2.weights
        );
        println!("({}) grad_lin2_biases = {:.10}", i, model_grad.lin2.biases);

        //println!("({}) weights = {:.10}", i, lin.weights);
        //println!("({}) biases = {:.10}", i, lin.biases);
        model = model.update(&mut optimizer, model_grad);
    }
    //let test = tensor!(1, 2 => [2.0, -5.0]);
    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);
}

pub fn train_relu(dataset: (Tensor<2>, Tensor<2>), two_layers: bool, epochs: usize, lr: f32) {
    let (datas, labels) = dataset;

    let mut model = ReLUModel::new(2, 1, two_layers);

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
        model = model.update(&mut optimizer, model_grad);
    }
    //let test = tensor!(1, 2 => [2.0, -5.0]);
    let test = datas.clone();
    let output = model.forward(&test);
    println!("output = {:.4}", output);
}
