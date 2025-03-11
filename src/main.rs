use ml_rs::{
    matrix,
    nn::nn_struct::{Backward, Forward, Linear, Loss, Update, MSE, SGD},
};

fn main() {
    let datas = matrix![
        3, 2, 1, 0, -1;
        3, 2, 1, 0, -1
    ];
    let labels = matrix![7, 5, 3, 1, -1];

    let (_, no_data) = labels.shape();

    let mut lin = Linear::new(2, 1);
    let mut optimizer = SGD::new(0.03);
    let mut loss_function = MSE {};

    const EPOCHS: usize = 40;
    for i in 0..EPOCHS {
        for j in 0..no_data {
            let data = datas.col(j);
            let label = labels.col(j);
            let output = lin.forward(&data);
            let loss = loss_function.loss(output.clone(), label.clone());
            let loss_grad = loss_function.loss_grad(output, label);
            let grad = lin.backward(&loss_grad);
            println!("({i}) loss = {:.6}", loss);
            lin = lin.update(&mut optimizer, grad);
        }
    }
    let test = matrix![
        4;
        -5
    ];
    let output = lin.forward(&test);
    println!("output = {:.4}", output);
}
