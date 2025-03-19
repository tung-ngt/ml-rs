use ml_rs::{
    examples::logic_gate,
    random::{pcg::PCG, RandomGenerator},
};

fn main() {
    let mut random_generator = PCG::new(42, 1);
    let mut random_fn = || random_generator.next_normal(0.0, 0.5);
    logic_gate::train_sigmoid(
        logic_gate::xor_dataset(),
        true,
        10 * 1000,
        1.0,
        Some(&mut random_fn),
    );
}
