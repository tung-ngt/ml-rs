use ml_rs::{
    examples::{logic_gate, simple_cnn::train},
    random::{pcg::PCG, RandomGenerator},
};

fn main() {
    //let mut random_generator = PCG::new(42, 6);
    //let mut random_fn = || random_generator.next_normal(0.0, 0.05);
    //logic_gate::train_relu(
    //    logic_gate::xor_dataset(),
    //    true,
    //    10 * 1000,
    //    0.01,
    //    Some(&mut random_fn),
    //);

    train();
}
