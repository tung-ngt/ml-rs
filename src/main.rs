use ml_rs::examples::logic_gate;

fn main() {
    logic_gate::train_sigmoid(logic_gate::xor_dataset(), true, 10 * 1000, 1.0);
}
