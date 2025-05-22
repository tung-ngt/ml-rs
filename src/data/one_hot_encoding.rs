use crate::tensor::Tensor;

pub fn one_hot_encoding(labels: Tensor<1>, no_classes: usize) -> Tensor<2> {
    let batch = labels.no_elements();
    let mut data = vec![0.0; batch * no_classes];
    let shape = [batch, no_classes];
    let strides = Tensor::get_strides(&shape);

    for b in 0..batch {
        let label = labels[b] as usize;
        data[b * strides[0] + label] = 1.0;
    }

    Tensor::with_data(&shape, &strides, 0, data.into())
}
