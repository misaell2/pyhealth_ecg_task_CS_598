"""
ECG Multi-Label Task Example with Dual Ablation Study.

This script demonstrates how to use the ECGMultiLabelTask and performs
a dual ablation study by varying:

1. Input length (timesteps)
2. Label set (task definition)

This highlights how both input configuration and task design affect
the resulting data representation.
"""

import numpy as np
from pyhealth.tasks.ecg_classification import ECGMultiLabelTask


def run_experiment(num_timesteps: int, labels: list) -> None:
    """
    Run experiment with given ECG length and label set.
    """

    task = ECGMultiLabelTask(labels=labels)

    patient = {
        "ecg": np.random.rand(num_timesteps, 12),
        "labels": ["AF", "RBBB"]
    }

    samples = task(patient)

    print("=" * 50)
    print(f"Timesteps: {num_timesteps}")
    print(f"Label set: {labels}")
    print(f"Input shape: {samples[0]['x'].shape}")
    print(f"Encoded labels: {samples[0]['y']}")
    print(f"Number of active labels: {np.sum(samples[0]['y'])}")
    print("=" * 50)


if __name__ == "__main__":
    # Ablation dimensions
    timesteps_list = [50, 100, 200]

    label_sets = [
        ["AF", "RBBB"],
        ["AF", "I-AVB", "RBBB"],
        ["AF", "I-AVB", "LBBB", "RBBB"],
    ]

    # Run full grid
    for t in timesteps_list:
        for labels in label_sets:
            run_experiment(t, labels)