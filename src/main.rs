use ml_rs::{
    nn::{
        layers::{linear::Linear, sigmoid::Sigmoid},
        loss::mse::{reduction, MSE},
        optimizer::sgd::SGD,
        Backward, Forward, Loss, Update,
    },
    tensor,
};

fn main() {
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

    let mut lin = Linear::new(2, 1);
    let mut sigmoid = Sigmoid::default();
    let mut optimizer = SGD::new(0.1);
    let mut loss_function = MSE::<reduction::Mean>::default();

    const EPOCHS: usize = 10 * 1000;
    for i in 0..EPOCHS {
        let lin_output = lin.forward(&datas);
        let output = sigmoid.forward(&lin_output);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let sigmoid_grad = sigmoid.backward(&loss_grad);
        let lin_grad = lin.backward(&sigmoid_grad.input_grad());
        println!("({}) loss = {:.10}", i, loss);
        //println!("({}) grad_weights = {:.10}", i, grad.weights);
        //println!("({}) grad_biases = {:.10}", i, grad.biases);
        //println!("({}) weights = {:.10}", i, lin.weights);
        //println!("({}) biases = {:.10}", i, lin.biases);
        lin = lin.update(&mut optimizer, lin_grad);
    }
    //let test = tensor!(1, 2 => [2.0, -5.0]);
    let test = datas.clone();
    let lin_output = lin.forward(&test);
    let output = sigmoid.forward(&lin_output);
    println!("output = {:.4}", output);
}
