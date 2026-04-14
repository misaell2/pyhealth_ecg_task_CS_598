# ECG Multi-Label Classification Task for PyHealth

## Overview
This project implements a standalone ECG multi-label classification task for PyHealth, inspired by:

**Nonaka & Seita (2021)**  
"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"  
https://proceedings.mlr.press/v149/nonaka21a.html

## Task Description
The task processes 12-lead ECG signals and predicts multiple cardiac conditions:

- Atrial Fibrillation (AF)
- First-degree AV Block (I-AVB)
- Left Bundle Branch Block (LBBB)
- Right Bundle Branch Block (RBBB)

### Input Format
Each patient record is expected to follow this structure:
```python
{
  "ecg": np.ndarray of shape (T, 12), # ECG signal
  "labels": List[str]              #  list of condition labels
}

## Repository Structure
pyhealth/
  tasks/
    ecg_classification.py   # Task implementation

tests/
  test_ecg_task.py          # Unit tests using synthetic data

examples/
  ecg_multilabel_resnet.py  # Example usage + ablation study

docs/
  ecg_task.md               # High-level documentation
  api/tasks/                # PyHealth-compatible .rst docs
