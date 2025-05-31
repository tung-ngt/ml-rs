use ml_rs::examples::{
    //mnist_classification,
    dynamic_mnist_classification,
};
fn main() {
    // compile model
    //mnist_classification::train();

    // dynamic model
    dynamic_mnist_classification::train();
}
