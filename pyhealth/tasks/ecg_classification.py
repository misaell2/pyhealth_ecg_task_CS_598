from typing import Dict, List, Any
import numpy as np


class ECGMultiLabelTask:
    """
    ECG Multi-label classification task.

    This task converts ECG signals into samples suitable for
    multi-label classification problems.
    """

    def __init__(self, labels: List[str]):
        """
        Initialize the task.

        Args:
            labels (List[str]): List of possible labels.
        """
        self.labels = labels

    def __call__(self, patient: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a patient record into model-ready samples.

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

        for i, label in enumerate(self.labels):
            if label in patient_labels:
                label_vector[i] = 1.0

        return label_vector