from typing import Dict, List, Any
import numpy as np

# Import BaseTask from PyHealth
from pyhealth.tasks.base_task import BaseTask


class ECGMultiLabelTask(BaseTask):
    """ECG Multi-label classification task.

    This task converts ECG signals into samples suitable for
    multi-label classification problems.
    """

    def __init__(self, labels: List[str]):
        """Initialize the task.

        Args:
            labels (List[str]): List of possible labels.
        """
        super().__init__()
        self.labels = labels

    def __call__(self, patient: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a patient record into model-ready samples.

        Args:
            patient (Dict[str, Any]): Patient data containing:
                - "ecg": ECG signal (T, 12)
                - "labels": list of labels

        Returns:
            List[Dict[str, Any]]: Processed samples
        """
        samples = []

        if "ecg" not in patient or "labels" not in patient:
            return samples

        ecg_signal = patient["ecg"]
        patient_labels = patient["labels"]

        label_vector = self._encode_labels(patient_labels)

        samples.append({
            "x": np.array(ecg_signal, dtype=np.float32),
            "y": label_vector,
        })

        return samples

    def _encode_labels(self, patient_labels: List[str]) -> np.ndarray:
        """Convert labels into multi-hot encoding."""
        label_vector = np.zeros(len(self.labels), dtype=np.float32)

        for label in patient_labels:
            if label in self.labels:
                idx = self.labels.index(label)
                label_vector[idx] = 1.0

        return label_vector