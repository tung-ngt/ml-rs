use std::{
    fs::File,
    io::{self, BufRead, BufReader},
};

use crate::tensor::Tensor;

#[derive(Debug)]
pub enum ReadCSVError {
    IOError(io::Error),
    ParseFloatError {
        row: usize,
        col: usize,
    },
    MismatchRowSize {
        row: usize,
        expected: usize,
        got: usize,
    },
}

impl From<io::Error> for ReadCSVError {
    fn from(value: io::Error) -> Self {
        Self::IOError(value)
    }
}

pub fn read_csv(filename: &str, skip_header: Option<usize>) -> Result<Tensor<2>, ReadCSVError> {
    let file = File::open(filename)?;
    let buf_reader = BufReader::new(file);

    let mut data = Vec::new();

    let mut prev_cols = 0;
    let mut no_rows = 0;
    for line in buf_reader.lines().skip(skip_header.unwrap_or(0)) {
        let line = line?;

        let elements = line.split(',');

        let mut no_cols = 0;
        for e in elements {
            let e: f32 = e.parse().map_err(|_| ReadCSVError::ParseFloatError {
                row: no_rows,
                col: no_cols,
            })?;
            data.push(e);
            no_cols += 1;
        }

        if no_rows == 0 {
            prev_cols = no_cols;
        }

        if prev_cols != no_cols {
            return Err(ReadCSVError::MismatchRowSize {
                row: no_rows,
                expected: prev_cols,
                got: no_cols,
            });
        }

        no_rows += 1;
    }

    let shape = [no_rows, prev_cols];
    let strides = Tensor::get_strides(&shape);
    Ok(Tensor::with_data(&shape, &strides, 0, data.into()))
}
