use core::panic;
use std::{env, fs, path::Path, time::Instant};

use crate::{
    data::{csv::read_csv, one_hot_encoding::one_hot_encoding},
    metric::classification::ClassificationReport,
    nn::{
        layers::{
            conv::Conv2D, linear::Linear, max_pool::MaxPool2D, pad::Pad2D, relu::ReLU,
            softmax::Softmax, stack::Stack, DynGrad,
        },
        loss::{cross_entropy::CrossEntropy, reduction},
        optimizer::{sgd::SGD, DynOptimizer},
        DynLayer, Loss,
    },
    random::{pcg::PCG, RandomGenerator},
    tensor::{
        conv::conv_output_shape,
        pad::{pad2d_output_size, PaddingSize, PaddingType},
        pooling::pooling_output_shape,
        Tensor,
    },
};

pub fn load_config(config_path: &str) -> (usize, Stack, DynOptimizer) {
    let mut stack: Vec<Box<dyn DynLayer>> = Vec::new();

    let config_string = fs::read_to_string(config_path)
        .unwrap_or_else(|_| panic!("cannot find config {}", config_path));

    let mut configs = config_string.split("####################\n");
    let input_shape = configs.next().expect("missing or wrong input shape config");
    let mut shape_slit = input_shape.trim().split(",");
    let batch_size = shape_slit
        .next()
        .expect("missing input shape")
        .parse()
        .expect("cannot parse input shape");
    let input_shape: [usize; 4] = [
        batch_size,
        shape_slit
            .next()
            .expect("missing input shape")
            .parse()
            .expect("cannot parse input shape"),
        shape_slit
            .next()
            .expect("missing input shape")
            .parse()
            .expect("cannot parse input shape"),
        shape_slit
            .next()
            .expect("missing input shape")
            .parse()
            .expect("cannot parse input shape"),
    ];

    let model_config = configs
        .next()
        .expect("missing or wrong model config")
        .trim();

    let mut random_generator = PCG::new(42, 6);
    let mut last_input_shape = input_shape;
    for line in model_config.lines() {
        let line = line.trim();
        let mut layer_configs = line.split(",");
        let layer_type = layer_configs.next().expect("cannot get layer type");
        match layer_type {
            "Pad2D" => {
                let pad_h: usize = layer_configs
                    .next()
                    .expect("missing pad h")
                    .parse()
                    .expect("failed to parse pad h");
                let pad_w: usize = layer_configs
                    .next()
                    .expect("missing pad w")
                    .parse()
                    .expect("failed to parse pad w");
                let padding = (PaddingSize::Same(pad_h), PaddingSize::Same(pad_w));
                let pad_value: f32 = layer_configs
                    .next()
                    .expect("missing pad value")
                    .parse()
                    .expect("failed to parse pad value");
                stack.push(Box::new(Pad2D::with_shape(
                    &input_shape,
                    padding.clone(),
                    PaddingType::Const(pad_value),
                )));
                last_input_shape = pad2d_output_size(&last_input_shape, padding);
            }
            "Conv2D" => {
                let c_in = layer_configs
                    .next()
                    .expect("missing channel in")
                    .parse()
                    .expect("cannot parse channel in");
                let c_out = layer_configs
                    .next()
                    .expect("missing channel out")
                    .parse()
                    .expect("cannot parse channel out");
                let k_h = layer_configs
                    .next()
                    .expect("missing kernel h")
                    .parse()
                    .expect("cannot parse kernel h");
                let k_w = layer_configs
                    .next()
                    .expect("missing kernel w")
                    .parse()
                    .expect("cannot parse kernel w");
                let stride_h = layer_configs
                    .next()
                    .expect("missing stride h")
                    .parse()
                    .expect("cannot parse stride h");
                let stride_w = layer_configs
                    .next()
                    .expect("missing stride w")
                    .parse()
                    .expect("cannot parse stride w");
                let dilation_h = layer_configs
                    .next()
                    .expect("missing dilation h")
                    .parse()
                    .expect("cannot parse dilation h");
                let dilation_w = layer_configs
                    .next()
                    .expect("missing dilation w")
                    .parse()
                    .expect("cannot parse dilation w");
                let mut random = || {
                    random_generator.next_normal(
                        0.0,
                        f32::sqrt(2.0 / (k_w as f32 * k_h as f32 * last_input_shape[3] as f32)),
                    )
                };
                stack.push(Box::new(Conv2D::random_init_with_shape(
                    &last_input_shape,
                    &[c_out, k_h, k_w, c_in],
                    (stride_h, stride_w),
                    (dilation_h, dilation_w),
                    &mut random,
                )));
                last_input_shape = conv_output_shape(
                    &last_input_shape,
                    &[c_out, k_h, k_w, c_in],
                    (stride_h, stride_w),
                )
            }
            "ReLU" => {
                stack.push(Box::new(ReLU::default()));
            }
            "MaxPool2D" => {
                let k_h = layer_configs
                    .next()
                    .expect("missing kernel h")
                    .parse()
                    .expect("cannot parse kernel h");
                let k_w = layer_configs
                    .next()
                    .expect("missing kernel w")
                    .parse()
                    .expect("cannot parse kernel w");
                let stride_h = layer_configs
                    .next()
                    .expect("missing stride h")
                    .parse()
                    .expect("cannot parse stride h");
                let stride_w = layer_configs
                    .next()
                    .expect("missing stride w")
                    .parse()
                    .expect("cannot parse stride w");
                let dilation_h = layer_configs
                    .next()
                    .expect("missing dilation h")
                    .parse()
                    .expect("cannot parse dilation h");
                let dilation_w = layer_configs
                    .next()
                    .expect("missing dilation w")
                    .parse()
                    .expect("cannot parse dilation w");
                stack.push(Box::new(MaxPool2D::with_shape(
                    &last_input_shape,
                    (k_h, k_w),
                    (stride_h, stride_w),
                    (dilation_h, dilation_w),
                )));
                last_input_shape = pooling_output_shape(
                    &last_input_shape,
                    (k_h, k_w),
                    (stride_h, stride_w),
                    (dilation_h, dilation_w),
                );
            }
            "Flatten" => {
                // do not need to do anything
            }
            "Linear" => {
                let in_features = layer_configs
                    .next()
                    .expect("missing in_features")
                    .parse()
                    .expect("cannot parse in_features");
                let out_features = layer_configs
                    .next()
                    .expect("missing out_features")
                    .parse()
                    .expect("cannot parse in_features");
                let mut random =
                    || random_generator.next_normal(0.0, f32::sqrt(2.0 / in_features as f32));
                stack.push(Box::new(Linear::random_init(
                    in_features,
                    out_features,
                    &mut random,
                )));
            }
            "Softmax" => {
                stack.push(Box::new(Softmax::default()));
            }
            _ => panic!("unknown layer {}", layer_type),
        }
    }

    let mut optim_configs = configs
        .next()
        .expect("missing or wrong optim config")
        .split(",");
    let optim_name = optim_configs.next().expect("invalid optimizer").trim();

    let optim = match optim_name {
        "SGD" => {
            let lr = optim_configs
                .next()
                .expect("missing sgd lr")
                .trim()
                .parse()
                .expect("cannot parse sgd lr");
            DynOptimizer::SGD(SGD::new(lr))
        }
        _ => panic!("invalide optimizer"),
    };
    (batch_size, Stack::new(stack), optim)
}

pub fn get_data(path: &str) -> (Tensor<2>, Tensor<1>) {
    assert!(Path::new(path).is_file(), "File {} does not exist", path);

    let data = read_csv(path, Some(1)).expect("should not be error when read csv");
    let x = data.submatrix(.., 1..);
    let y = data.col(0).squeeze(1);
    (x, y)
}

pub fn train() {
    let args: Vec<String> = env::args().collect();
    let config_path = &args[1];
    let train_data_path = &args[2];
    let test_data_path = &args[3];
    let epochs: usize = args[4].parse().expect("invalid epoch");
    let (batch_size, mut model, mut optimizer) = load_config(config_path);
    let mut loss_fn = CrossEntropy::<reduction::Mean>::default();

    let (data, label) = get_data(train_data_path);
    let label_one_hot = one_hot_encoding(label.clone(), 10);
    let mean = data.mean();
    let std = data.std();
    let data = &(&data - mean) / std;

    let no_sample = data.shape()[0];
    let no_batch = no_sample.div_ceil(batch_size);

    let mut random = PCG::new(42, 6);
    for e in 0..epochs {
        let shuffled_index = random.next_shuffle(no_sample);
        let shuffled_data = data.shuffle_batch(&shuffled_index);
        let shuffled_label = label_one_hot.shuffle_batch(&shuffled_index);

        let start = Instant::now();
        let mut avg_loss = 0.0;
        for b in 0..no_batch {
            let batch_range = b * batch_size..usize::min((b + 1) * batch_size, no_sample);
            let batch_data = shuffled_data.subtensor(&[batch_range.clone(), 0..28 * 28]);
            let batch_label_one_hot = shuffled_label.submatrix(batch_range, ..);

            let predict = model.forward(&batch_data);
            let batch_loss = loss_fn.loss(predict.clone(), batch_label_one_hot.clone());

            let loss_grad = loss_fn.loss_grad(predict, batch_label_one_hot);
            let model_grad = model.backward(&loss_grad);
            model.update(&mut optimizer, &model_grad);

            avg_loss += batch_loss;
            println!(
                "epoch {}, batch ({}/{}): loss {}",
                e, b, no_batch, batch_loss
            );
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

    let (test_data, test_label) = get_data(test_data_path);
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
