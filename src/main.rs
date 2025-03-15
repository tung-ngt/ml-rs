use ml_rs::{
    nn::{
        layers::linear::Linear,
        loss::mse::{reduction, MSE},
        optimizer::sgd::SGD,
        Backward, Forward, Loss, Update,
    },
    tensor,
};

fn main() {
    let datas = tensor!(5, 2 => [
        [9.0, 1.0],
        [2.0, 6.0],
        [1.0, 3.0],
        [2.0, 3.0],
        [-1.0, -1.0]
    ]);
    let labels = tensor!(5, 1 => [
        10.0,
        8.0,
        4.0,
        5.0,
        -2.0
    ]);

    //let &[no_data, _] = labels.shape();

    let mut lin = Linear::new(2, 1);
    let mut optimizer = SGD::new(0.01);
    let mut loss_function = MSE::<reduction::Mean>::default();

    const EPOCHS: usize = 1000;
    for i in 0..EPOCHS {
        let output = lin.forward(&datas);
        let loss = loss_function.loss(output.clone(), labels.clone());
        let loss_grad = loss_function.loss_grad(output, labels.clone());
        let grad = lin.backward(&loss_grad);
        println!("({}) loss = {:.10}", i, loss);
        //println!("({}) grad_weights = {:.10}", i, grad.weights);
        //println!("({}) grad_biases = {:.10}", i, grad.biases);
        //println!("({}) weights = {:.10}", i, lin.weights);
        //println!("({}) biases = {:.10}", i, lin.biases);
        lin = lin.update(&mut optimizer, grad);
    }
    let test = tensor!(1, 2 => [2.0, -5.0]);
    let output = lin.forward(&test);
    println!("output = {:.4}", output);
}
