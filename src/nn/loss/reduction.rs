pub trait Reduction {}

#[derive(Default)]
pub struct Mean;

#[derive(Default)]
pub struct Sum;

#[derive(Default)]
pub struct NoReduction;

#[derive(Default)]
pub struct MeanBatch;

#[derive(Default)]
pub struct MeanFeature;

impl Reduction for Mean {}
impl Reduction for Sum {}
impl Reduction for NoReduction {}
impl Reduction for MeanBatch {}
impl Reduction for MeanFeature {}
