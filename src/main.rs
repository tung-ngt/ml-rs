use ml_rs::{
    matrix,
    nn::nn_struct::{Backward, Forward, Linear, Loss, Update, MSE, SGD},
};

fn main() {
    let datas = matrix![
        9, 1;
        2, 6;
        1, 3;
        2, 3;
        -1, -1
    ];
    let labels = matrix![
        9, 1;
        2, 6;
        1, 3;
        2, 3;
        -1, -1
    ];

    //let labels = matrix![
    //    2;
    //    4;
    //    6;
    //    0;
    //    -2
    //];

    let (no_data, _) = labels.shape();

    let mut lin = Linear::new(2, 2);
    let mut optimizer = SGD::new(0.001);
    let mut loss_function = MSE {};

    const EPOCHS: usize = 10 * 1000;
    for i in 0..EPOCHS {
        for j in 0..no_data {
            let data = datas.row(j);
            let label = labels.row(j);
            let output = lin.forward(&data);
            let loss = loss_function.loss(output.clone(), label.clone());
            let loss_grad = loss_function.loss_grad(output, label);
            let grad = lin.backward(&loss_grad);
            //println!("({}) loss = {:.10}", i, loss);
            //println!("({}) grad_weights = {:.10}", i, grad.weights);
            //println!("({}) grad_biases = {:.10}", i, grad.biases);
            //println!("({}) weights = {:.10}", i, lin.weights);
            //println!("({}) biases = {:.10}", i, lin.biases);
            lin = lin.update(&mut optimizer, grad);
        }
    }
    let test = matrix![2, -5];
    let output = lin.forward(&test);
    println!("output = {:.4}", output);
}
