use super::Matrix;
use std::fmt;

fn print_matrix(matrix: &Matrix, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let precision = f.precision().unwrap_or(2);
    let padding = f.width().unwrap_or(0);
    let padding = " ".repeat(padding);
    let (rows, cols) = matrix.shape();

    writeln!(f, "[")?;
    for i in 0..rows {
        for j in 0..cols {
            write!(
                f,
                "{padding}  {cell:.precision$} ",
                cell = matrix[(i, j)],
                padding = padding,
                precision = precision
            )?;
        }
        writeln!(f)?;
    }
    writeln!(f, "{padding}] ({}x{})", rows, cols)?;
    Ok(())
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_matrix(self, f)
    }
}
