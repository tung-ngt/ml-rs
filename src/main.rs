use ml_rs::examples::{
    //logic_gate,
    dynamic_simple_cnn::train,
    //simple_cnn::train
};

fn main() {
    //let mut random_generator = PCG::new(32, 6);
    //let mut random_fn = || random_generator.next_normal(0.0, 0.05);
    //logic_gate::train_sigmoid(
    //    logic_gate::xor_dataset(),
    //    true,
    //    10 * 1000,
    //    10.0,
    //    Some(&mut random_fn),
    //);

    train();
}
