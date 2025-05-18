use super::{conv::PaddingSize, Tensor};
use std::fmt;

fn print_matrix(matrix: &Tensor<2>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let precision = f.precision().unwrap_or(2);
    let padding = f.width().unwrap_or(0);
    let padding = " ".repeat(padding);
    let &[rows, cols] = matrix.shape();

    writeln!(f, "[")?;
    for i in 0..rows {
        write!(f, "{padding}  ")?;
        for j in 0..cols {
            write!(
                f,
                "{cell:.precision$} ",
                cell = matrix[(i, j)],
                precision = precision
            )?;
        }
        writeln!(f)?;
    }
    write!(f, "{padding}] ({}x{})", rows, cols)?;
    Ok(())
}

fn print_vector(vector: &Tensor<1>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let precision = f.precision().unwrap_or(2);
    let padding = f.width().unwrap_or(0);
    let padding = " ".repeat(padding);
    let &[rows] = vector.shape();

    writeln!(f, "[")?;
    for i in 0..rows {
        writeln!(
            f,
            "{padding}  {cell:.precision$}",
            padding = padding,
            cell = vector[i],
            precision = precision
        )?;
    }
    write!(f, "{padding}] ({},)", rows)?;
    Ok(())
}

fn print_tensor3(matrix: &Tensor<3>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let precision = f.precision().unwrap_or(2);
    let padding = f.width().unwrap_or(0);
    let padding = " ".repeat(padding);
    let &[rows, cols, channels] = matrix.shape();

    writeln!(f, "[")?;
    for i in 0..rows {
        write!(f, "{padding}  ")?;
        for j in 0..cols {
            write!(f, "[")?;
            for k in 0..channels {
                write!(
                    f,
                    "{cell:.precision$} ",
                    cell = matrix[(i, j, k)],
                    precision = precision
                )?;
            }
            write!(f, "]")?;
        }
        writeln!(f)?;
    }
    write!(f, "{padding}] ({}x{}x{})", rows, cols, channels)?;
    Ok(())
}

impl fmt::Display for Tensor<2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_matrix(self, f)
    }
}

impl fmt::Display for Tensor<1> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_vector(self, f)
    }
}

impl fmt::Display for Tensor<3> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_tensor3(self, f)
    }
}

pub fn pad2d_same_size(
    image_size: (usize, usize),
    kernel_size: (usize, usize),
    strides: (usize, usize),
) -> (PaddingSize, PaddingSize) {
    let height_padding = image_size.0 * (strides.0 - 1) + kernel_size.0 - strides.0;
    let height_padding = if height_padding % 2 == 0 {
        PaddingSize::Same(height_padding / 2)
    } else {
        PaddingSize::Diff(height_padding / 2, height_padding / 2 + 1)
    };

    let width_padding = image_size.1 * (strides.1 - 1) + kernel_size.1 - strides.1;
    let width_padding = if width_padding % 2 == 0 {
        PaddingSize::Same(width_padding / 2)
    } else {
        PaddingSize::Diff(width_padding / 2, width_padding / 2 + 1)
    };

    (height_padding, width_padding)
}

pub fn pad2d_full_size(kernel_size: (usize, usize)) -> (PaddingSize, PaddingSize) {
    let height_padding = PaddingSize::Same(kernel_size.0 - 1);
    let width_padding = PaddingSize::Same(kernel_size.1 - 1);

    (height_padding, width_padding)
}

pub fn conv_output_size(
    images_size: &[usize; 4],
    kernels_size: &[usize; 4],
    strides: (usize, usize),
) -> [usize; 4] {
    let &[b, h, w, _c_in] = images_size;

    let &[c_out, k_h, k_w, _] = kernels_size;
    let new_h = (h - k_h) / strides.0 + 1;
    let new_w = (w - k_w) / strides.1 + 1;
    [b, new_w, new_h, c_out]
}

pub fn pooling_output_size(
    image_size: &[usize; 4],
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilation: (usize, usize),
) -> [usize; 4] {
    let &[b, h, w, c] = image_size;
    let h_out = (h - dilation.0 * (kernel_size.0 - 1) - 1) / strides.0 + 1;
    let w_out = (w - dilation.1 * (kernel_size.1 - 1) - 1) / strides.1 + 1;
    [b, h_out, w_out, c]
}

pub fn flatten_output_shape<const INPUT_DIMENSIONS: usize, const OUTPUT_DIMENSIONS: usize>(
    input_shape: &[usize; INPUT_DIMENSIONS],
    start: Option<usize>,
    stop: Option<usize>,
) -> [usize; OUTPUT_DIMENSIONS] {
    let mut new_shape = [1; OUTPUT_DIMENSIONS];

    let start = start.unwrap_or(0);
    let stop = stop.unwrap_or(INPUT_DIMENSIONS);

    let flatten_dims = start..stop;

    let mut n = OUTPUT_DIMENSIONS;
    for (i, s) in input_shape.iter().enumerate().rev() {
        if !flatten_dims.contains(&i) || i == INPUT_DIMENSIONS - 1 {
            n -= 1;
        }
        new_shape[n] *= *s;
    }
    new_shape
}

pub fn conv_unused_inputs(
    image_size: (usize, usize),
    kernel_size: (usize, usize),
    strides: (usize, usize),
) -> (usize, usize) {
    let [_, h_out, w_out, _] = conv_output_size(
        &[1, image_size.0, image_size.1, 1],
        &[1, kernel_size.0, kernel_size.1, 1],
        strides,
    );
    let unused_h = image_size.0 - (h_out - 1) * strides.0 - kernel_size.0;
    let unused_w = image_size.1 - (w_out - 1) * strides.1 - kernel_size.1;

    (unused_h, unused_w)
}

#[cfg(test)]
mod tensor_display_tests {
    use crate::tensor::Tensor;

    #[test]
    fn print_matrix() {
        let a = Tensor::identity(3);
        let expected_string = "[\n  1.00 0.00 0.00 \n  0.00 1.00 0.00 \n  0.00 0.00 1.00 \n] (3x3)";
        let formatted_string = format!("{}", a);
        assert!(
            expected_string == formatted_string,
            "Wrong format expeted:\n{}\ngot:\n{}",
            expected_string,
            formatted_string
        );
        //println!("a = {}", a);
    }

    #[test]
    fn print_vector() {
        let a = Tensor::vector_filled(5, 2f32);
        let expected_string = "[\n  2.00\n  2.00\n  2.00\n  2.00\n  2.00\n] (5,)";
        let formatted_string = format!("{}", a);
        assert!(
            expected_string == formatted_string,
            "Wrong format expeted:\n{}\ngot:\n{}",
            expected_string,
            formatted_string
        );
        //println!("a = {}", a);
    }
}
