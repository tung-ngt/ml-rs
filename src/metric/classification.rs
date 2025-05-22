use crate::tensor::Tensor;

pub struct ClassificationReport {
    confusion_matrix: Tensor<2>,
    precision: Tensor<1>,
    recall: Tensor<1>,
    f1: Tensor<1>,
    accuracy: f32,
}

impl ClassificationReport {
    pub fn new(no_classes: usize, prediction: Tensor<1>, target: Tensor<1>) -> Self {
        let confusion_matrix = Self::compute_confusion_matrix(no_classes, &prediction, &target);
        let precision = Self::compute_precision(&confusion_matrix);
        let recall = Self::compute_recall(&confusion_matrix);
        let f1 = Self::compute_f1(&precision, &recall);
        let accuracy = Self::compute_accuracy(&confusion_matrix);
        Self {
            confusion_matrix,
            precision,
            recall,
            f1,
            accuracy,
        }
    }

    ///Return confusion matrix target x prediction. Target is on the vertical axis
    ///and prediction is on the horizontal axis
    pub fn compute_confusion_matrix(
        no_classes: usize,
        prediction: &Tensor<1>,
        target: &Tensor<1>,
    ) -> Tensor<2> {
        assert!(
            prediction.no_elements() == target.no_elements(),
            "Cannot compute confusion matrix for {} predictions and {} targets",
            prediction.no_elements(),
            target.no_elements()
        );

        let mut data = vec![0.0; no_classes * no_classes];
        for i in 0..target.no_elements() {
            let actual = target[i] as usize;
            let predicted = prediction[i] as usize;
            data[actual * no_classes + predicted] += 1.0;
        }

        Tensor::matrix_with_data(no_classes, no_classes, &[no_classes, 1], 0, data.into())
    }
    pub fn compute_precision(confusion_matrix: &Tensor<2>) -> Tensor<1> {
        let tp = confusion_matrix.diag();
        let predicted = confusion_matrix.sum_row();
        tp.div_elem(&predicted)
    }

    pub fn compute_recall(confusion_matrix: &Tensor<2>) -> Tensor<1> {
        let tp = confusion_matrix.diag();
        let actual = confusion_matrix.sum_col();
        tp.div_elem(&actual)
    }

    pub fn compute_f1(precision: &Tensor<1>, recall: &Tensor<1>) -> Tensor<1> {
        precision
            .mul_elem(recall)
            .scale(2.0)
            .div_elem(&(precision + recall))
    }

    pub fn compute_accuracy(confusion_matrix: &Tensor<2>) -> f32 {
        let tp = confusion_matrix.diag().sum();
        let sum = confusion_matrix.sum();
        tp / sum
    }

    ///Return confusion matrix target x prediction. Target is on the vertical axis
    ///and prediction is on the horizontal axis
    pub fn confusion_matrix(&self) -> &Tensor<2> {
        &self.confusion_matrix
    }

    pub fn precision(&self) -> &Tensor<1> {
        &self.precision
    }

    pub fn recall(&self) -> &Tensor<1> {
        &self.recall
    }

    pub fn f1(&self) -> &Tensor<1> {
        &self.f1
    }

    pub fn accuracy(&self) -> f32 {
        self.accuracy
    }
}
