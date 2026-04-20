"""
Unit tests for ECGMultiLabelCardiologyTask.

These tests validate the functionality of the ECG multi-label classification
task, ensuring correct input processing, label encoding, and handling of
edge cases such as missing data or empty labels.

All tests use synthetic data
"""
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import pytest #for parameterization
from pyhealth.tasks.ecg_classification import ECGMultiLabelCardiologyTask

def test_ecg_task_basic():
    """
    Test basic functionality of ECGMultiLabelCardiologyTask.

    This test verifies that:
    - A valid patient record produces exactly one sample
    - The ECG signal is correctly passed through as input (x)
    - The label vector (y) has the correct shape
    - Labels are correctly encoded into a multi-hot vector

    Expected behavior:
        - Input ECG shape is preserved
        - Output label vector matches the provided labels
    """
    task = ECGMultiLabelCardiologyTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

    patient = {
        "ecg": np.random.rand(100, 12),  # synthetic ECG signal
        "labels": ["AF", "RBBB"]         # two active labels
    }

    samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["x"].shape == (100, 12)
    assert samples[0]["y"].shape == (4,)
    assert samples[0]["y"][0] == 1  # AF
    assert samples[0]["y"][3] == 1  # RBBB


@pytest.mark.parametrize("timesteps", [50, 100, 200])
def test_ecg_task_comprehensive(timesteps):
    """
    Test basic functionality of ECGMultiLabelCardiologyTask across multiple signal lengths.

    This test verifies that:
    - A valid patient record produces exactly one sample regardless of signal length
    - The ECG signal shape is maintained across different timestamps
    - The label vector (y) has the correct shape
    - Labels are correctly encoded into a multi-hot vector

    Expected behavior:
        - Input ECG shape is preserved through different signal lengths
        - Output sample matches the input
    """
    task = ECGMultiLabelTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

    patient = {
        "ecg": np.random.rand(timesteps, 12),  # synthetic ECG signal
        "labels": ["AF", "RBBB"]         # two active labels
    }

    samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["x"].shape == (timesteps, 12)
    assert samples[0]["y"].shape == (4,)
    assert samples[0]["y"][0] == 1  # AF
    assert samples[0]["y"][3] == 1  # RBBB


def test_empty_patient():
    """
    Test behavior when patient data is missing required fields.

    This test ensures that:
    - If the patient dictionary does not contain required keys
      ("ecg" or "labels"), the task returns an empty list
    - No errors are raised for incomplete input

    Expected behavior:
        - Output is an empty list
    """
    task = ECGMultiLabelCardiologyTask(labels=["AF"])

    patient = {}  # missing both "ecg" and "labels"
    samples = task(patient)

    assert samples == []


def test_no_labels():
    """
    Test behavior when patient has no labels.

    This test verifies that:
    - A patient with valid ECG data but no labels still produces a sample
    - The output label vector is all zeros (no active conditions)

    Expected behavior:
        - Output sample exists
        - Label vector contains only zeros
    """
    task = ECGMultiLabelCardiologyTask(labels=["AF", "LBBB"])

    patient = {
        "ecg": np.random.rand(50, 12),  # shorter ECG signal
        "labels": []                    # no conditions present
    }

    samples = task(patient)

    assert np.all(samples[0]["y"] == 0)
    
def test_different_label_sets():
    """
    Test task behavior under different label configurations.
    """
    label_sets = [
        ["AF", "RBBB"],
        ["AF", "I-AVB", "RBBB"],
        ["AF", "I-AVB", "LBBB", "RBBB"],
    ]

    patient = {
        "ecg": np.random.rand(100, 12),
        "labels": ["AF", "RBBB"]
    }

    for labels in label_sets:
        task = ECGMultiLabelCardiologyTask(labels=labels)
        samples = task(patient)

        assert samples[0]["y"].shape == (len(labels),)
