"""
ECG Multi-Label Task Example with Simple Ablation Study.

This script demonstrates how to use the ECGMultiLabelTask from PyHealth
and performs a simple ablation study by varying the length of the input
ECG signal. (simple ablation, can work on making a more extensive ablation)

Ablation Study:
---------------
We evaluate how different input lengths (timesteps) affect the processed
input shape and label encoding. This example highlights how input 
configuration changes propagate through the task.

All data used here is synthetic for fast execution and reproducibility.
"""

import numpy as np
from pyhealth.tasks.ecg_classification import ECGMultiLabelTask


def run_experiment(num_timesteps: int) -> None:
    """
    Runs a single experiment with a specified ECG length.

    This function:
    - Creates a synthetic ECG signal with a given number of timesteps
    - Applies the ECGMultiLabelTask
    - Prints the resulting input shape and encoded labels

    Args:
        num_timesteps (int): Number of time steps in the ECG signal.

    Returns:
        None
    """
    task = ECGMultiLabelTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

    # Create synthetic patient data
    patient = {
        "ecg": np.random.rand(num_timesteps, 12),  # ECG signal: (T, 12 leads)
        "labels": ["AF"]  # single active label
    }

    samples = task(patient)

    # Display results
    print("=" * 40)
    print(f"Experiment with {num_timesteps} timesteps")
    print(f"Input shape: {samples[0]['x'].shape}")
    print(f"Encoded labels: {samples[0]['y']}")
    print("=" * 40)
    
    active_labels = np.sum(samples[0]["y"])
    print(f"Number of active labels: {active_labels}")


if __name__ == "__main__":
    """
    Main entry point for running ablation experiments.

    We test multiple ECG lengths to observe how the task processes
    varying input sizes.
    """
    for t in [50, 100, 200]:
        run_experiment(t)