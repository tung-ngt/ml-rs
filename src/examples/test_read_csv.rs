use std::path::Path;

use crate::data::csv::read_csv;

pub fn test_read_csv() {
    assert!(
        Path::new("data/optdigits.tes").is_file(),
        "Assume that there is data/optdigits.tes"
    );

    let a = read_csv("data/optdigits.tes", None)
        .expect("should not be error when read file data/optdigits.tes");

    println!("{:?}", a.shape());
}
