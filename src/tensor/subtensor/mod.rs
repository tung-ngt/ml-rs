pub mod squeeze;
pub mod submatrix;
pub mod subtensor;
pub mod subvector;

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub trait TensorIndex {
    fn bound(&self) -> (Option<usize>, Option<usize>);
}

impl TensorIndex for usize {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(*self), Some(*self + 1))
    }
}

impl TensorIndex for Range<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(self.start), Some(self.end))
    }
}

impl TensorIndex for RangeFrom<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(self.start), None)
    }
}

impl TensorIndex for RangeFull {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, None)
    }
}

impl TensorIndex for RangeInclusive<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (Some(*self.start()), Some(*self.end() + 1))
    }
}

impl TensorIndex for RangeTo<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, Some(self.end))
    }
}

impl TensorIndex for RangeToInclusive<usize> {
    fn bound(&self) -> (Option<usize>, Option<usize>) {
        (None, Some(self.end + 1))
    }
}
