#[macro_export]
macro_rules! matrix {
    ($(
        $( $elements:expr),+
    );+) => {{
        const ROWS: usize = matrix!(@count_rows; $($($elements),*);+);
        const COLS: usize = matrix!(@count_cols; $($($elements),*);+);
        const NO_ELEMENTS: usize = matrix!(@count_elements; $($($elements),*);+);
        const {
            assert!(ROWS * COLS == NO_ELEMENTS, "Rows must have the same number of elements. So do columns")
        }
        let mut data: Vec<f32> = Vec::with_capacity(NO_ELEMENTS);

        $(
            $(
                data.push($elements as f32);
            )+
        )+

        $crate::matrix::Matrix::with_data(ROWS, COLS, COLS, 0, std::sync::Arc::from(data))
    }};
    (@count_rows; $($($elements:expr),+);+ ) => (0usize $(+ { $(let _ = $elements;)+; 1})+);
    (@count_cols; $($first_row_elems:expr),+) => (0usize $(+ {let _ = $first_row_elems; 1})+);
    (@count_cols; $($first_row_elems:expr),+; $($($rows:expr),+);+ ) => (0usize $(+ {let _ = $first_row_elems; 1})+);
    (@count_elements; $($($elements:expr),+);+ ) => (0usize $(+ (0usize $( + {let _ = $elements; 1})+))+);
}
