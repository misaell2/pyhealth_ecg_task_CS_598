from typing import Dict, List, Any, Optional, Union
import os
import numpy as np

from scipy.io import loadmat
from pyhealth.tasks.base_task import BaseTask  # Import BaseTask from PyHealth


class ECGMultiLabelCardiologyTask(BaseTask):
    """ECG Multi-label classification task.

    This task converts ECG signals into samples suitable for
    multi-label classification problems.
    """
    task_name: str = "ECGMultiLabelCardiologyTask"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multilabel"}

    def __init__(
        self,
        labels: List[str],
        epoch_sec: int = 10,
        shift: int = 5,
        sampling_rate: int = 500,
        **kwargs,
    ):

        """Initialize the task.

        Args:
            labels (List[str]): List of possible labels.
        """
        super().__init__(**kwargs)
        self.labels = labels
        self.epoch_sec = epoch_sec
        self.shift = shift
        self.sampling_rate = sampling_rate
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}

    def __call__(self, patient: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a patient record into model-ready samples.

        Args:
            patient (Dict[str, Any]): Patient data containing:
                - "ecg": ECG signal (T, 12)
                - "labels": list of labels

        Returns:
            List[Dict[str, Any]]: Processed samples
        """
        visits = self._normalize_input(patient)
        samples = []

        window_size = self.sampling_rate * self.epoch_sec
        step_size = self.sampling_rate * self.shift

        for visit in visits:
            if not self._is_valid_visit(visit):
                continue

            root = visit["load_from_path"]
            patient_id = visit["patient_id"]
            signal_file = visit["signal_file"]
            label_file = visit["label_file"]

            signal_path = os.path.join(root, signal_file)
            label_path = os.path.join(root, label_file)

            signal = self._load_signal(signal_path)
            if signal is None:
                continue

            metadata = self._parse_header_metadata(label_path)
            dx_codes = metadata["dx_codes"]
            sex = metadata["sex"]
            age = metadata["age"]

            label_vector = self._encode_labels(dx_codes)

            # Expected signal shape from CardiologyDataset-style .mat files: (leads, timesteps)
            if signal.ndim != 2 or signal.shape[1] < window_size:
                continue

            num_windows = (signal.shape[1] - window_size) // step_size + 1

            visit_id = os.path.splitext(os.path.basename(signal_file))[0]

            for index in range(num_windows):
                start = index * step_size
                end = start + window_size
                signal_window = signal[:, start:end].astype(np.float32)

                samples.append(
                    {
                        "patient_id": patient_id,
                        "visit_id": visit_id,
                        "record_id": len(samples) + 1,
                        "signal": signal_window,
                        "label": label_vector.copy(),
                        "Sex": sex,
                        "Age": age,
                    }
                )

        return samples

    def _normalize_input(
        self,
        patient: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize input into a list of visit dicts."""
        if isinstance(patient, list):
            return patient
        if isinstance(patient, dict):
            return [patient]
        return []


    def _is_valid_visit(self, visit: Dict[str, Any]) -> bool:
        """Checks whether the visit has the minimum required fields."""
        required_keys = {"load_from_path", "patient_id", "signal_file", "label_file"}
        return required_keys.issubset(visit.keys())


    def _load_signal(self, signal_path: str) -> Optional[np.ndarray]:
        """Load ECG signal from a .mat file."""
        try:
            mat = loadmat(signal_path)
        except Exception:
            return None

        # PyHealth's older built-in cardiology tasks use mat["val"].
        signal = mat.get("val")
        if signal is None:
            return None

        return np.asarray(signal, dtype=np.float32)


    def _parse_header_metadata(self, header_path: str) -> Dict[str, List[str]]:
        """Parse #Dx, #Sex, and #Age from a PhysioNet-style .hea header."""
        dx_codes: List[str] = []
        sex: List[str] = []
        age: List[str] = []

        try:
            with open(header_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()

                    if line.startswith("#Dx:"):
                        value = line.split(":", 1)[1].strip()
                        dx_codes = [x.strip() for x in value.split(",") if x.strip()]

                    elif line.startswith("#Sex:"):
                        value = line.split(":", 1)[1].strip()
                        sex = [value] if value else []

                    elif line.startswith("#Age:"):
                        value = line.split(":", 1)[1].strip()
                        age = [value] if value else []
        except Exception:
            pass

        return {
            "dx_codes": dx_codes,
            "sex": sex,
            "age": age,
        }


    def _encode_labels(self, patient_labels: List[str]) -> np.ndarray:
        """Convert labels into multi-hot encoding."""
        label_vector = np.zeros(len(self.labels), dtype=np.float32)

        for label in patient_labels:
            if label in self.labels:
                idx = self.labels.index(label)
                label_vector[idx] = 1.0

        return label_vector
