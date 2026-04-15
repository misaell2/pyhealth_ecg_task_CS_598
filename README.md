# ECG Multi-Label Classification Task for PyHealth

## Overview
This project implements a standalone ECG multi-label classification task for PyHealth, inspired by:

**Nonaka & Seita (2021)**  
"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"  
https://proceedings.mlr.press/v149/nonaka21a.html

---
## Task Description
The task processes 12-lead ECG signals and predicts multiple cardiac conditions simultaneously:

- Atrial Fibrillation (**AF**)
- First-degree AV Block (**I-AVB**)
- Left Bundle Branch Block (**LBBB**)
- Right Bundle Branch Block (**RBBB**)
---
## Ablation Study

In this repository, we perform an ablation study along two key dimensions:

### 1. **Input Configuration (ECG Length)**
   We vary the number of timesteps in the ECG signal (e.g., 50, 100, 200) to observe how input size affects the processed representation. This corresponds to a *data-level* or preprocessing variation.

### 2. **Task Configuration (Label Set)**
   We vary the set of diagnostic labels used for classification (e.g., 2-label vs 4-label setups). This changes the dimensionality and complexity of the output space, effectively altering the learning problem itself. This corresponds to a *problem-level* variation.
   
---

### Comparison to Nonaka & Seita's paper (2021)
While the original paper evaluates performance across different models for a fixed task, our work in this repository explores how changes in the task definition itself, particularly the label space, impacts the structure of the input-output mapping.
Varying the label set is especially significant because it directly affects:
- the number of prediction targets
- the complexity of the classification problem
- the structure of the model output

  We provide a complementary perspective to Nonaka's work, highlighting that not only model choice, but also **task design**, plays an important role in ECG-based machine learning workflows. Below we explain the workflow:

---

### Input Format
Each patient record is expected to follow this structure:
```python
{
  "ecg": np.ndarray of shape (T, 12), # ECG signal
  "labels": List[str]              #  list of condition labels
}
```



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

========================================= 4 passed in 1.19s ==========================================
```
