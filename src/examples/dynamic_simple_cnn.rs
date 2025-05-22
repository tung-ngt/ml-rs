use crate::{
    nn::{
        layers::{conv::Conv2D, flatten::Flatten, stack::Stack},
        loss::{mse::MSE, reduction},
        optimizer::{sgd::SGD, DynOptimizer},
        DynLayer, Loss,
    },
    tensor,
    tensor::pad::{pad2d_same_size, PaddingType},
};

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

    let label = tensor!(1, 9, 2 => [
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

    let data = data.pad2d(pad2d_same_size((3, 3), (3, 3), (1, 1)), PaddingType::Zero);
    let data = data.flatten(Some(1), None);

    let layers: Vec<Box<dyn DynLayer>> = vec![
        Box::new(Conv2D::with_shape(
            &[1, 5, 5, 1],
            &[2, 3, 3, 1],
            (1, 1),
            (1, 1),
        )),
        Box::new(Flatten::<4>::new(Some(1), None)),
    ];
    let mut model = Stack::new(layers);

    let epochs = 500;
    let lr = 0.1;

    let mut sgd = DynOptimizer::SGD(SGD::new(lr));
    let mut loss_function = MSE::<reduction::Mean>::default();

    for _e in 0..epochs {
        let predict = model.forward(&data);
        let predict = predict.reshape(&[1, 9, 2]);
        let loss = loss_function.loss(predict.clone(), label.clone());
        println!("loss: {}", loss);
        let loss_grad = loss_function.loss_grad(predict, label.clone());
        let loss_grad = loss_grad.flatten(Some(1), Some(2));
        let model_grad = model.backward(&loss_grad);

        model.update(&mut sgd, &model_grad);
    }

    let predict = model.forward(&data);
    println!("{:.5}", predict);
}
