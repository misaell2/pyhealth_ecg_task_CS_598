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
```
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


  ### Creating Environment

  To create a virtual environment and install dependencies run the following in the main folder:
  ```python
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
```
### Running Tests
To ensure reproducibility and run the test files:
```python
export PYTHONPATH=$(pwd)
pytest tests/
```
Expected output:
```python
======================================== test session starts =========================================
platform linux -- Python 3.8.10, pytest-8.3.5, pluggy-1.5.0
rootdir: /<PATH TO REPO>/pyhealth_ecg_task_CS_598
collected 3 items

tests/test_ecg_task.py ...                                                                     [100%]

========================================= 3 passed in 1.19s ==========================================
```
