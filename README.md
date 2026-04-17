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

Utilizing PyHealth's BaseTask framework we are able to utilize many of it's features created for a wide range of medical data. However, becasue the paper we are trying to implement is specific to ECG we modified BaseTask by implementing a specific __call__ logic and schema that will enable PyHealth's learning pipelines to understand the 12-lead ECG singnals as matrices.

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
pip install numpy pytest polars
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
collected 7 items

tests/test_ecg_task.py ...                                                                     [100%]

========================================= 7 passed in 1.19s ==========================================
```



## Example Usage
For all example test cases,we import our created class ECGMultiLabelTask which inherits directly from PyHealth's BaseTask, however we did modify it to handle ECG specific data.

#basic manual test case

```python
from pyhealth.tasks.ecg_classification import ECGMultiLabelTask
import numpy as np

task = ECGMultiLabelTask(labels=["AF", "RBBB"])

patient = {
    "ecg": np.random.rand(100, 12),
    "labels": ["AF"]
}

samples = task(patient)
print(samples)
```

#Input Configuration (ECG Length) test case - processing signals of different lengths (50 vs 100 vs 200)
```python
from pyhealth.tasks.ecg_classification import ECGMultiLabelTask
import numpy as np

task = ECGMultiLabelTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

patient_50 = {
    "ecg": np.random.rand(50, 12),
    "labels": ["AF"]
}

patient_100 = {
    "ecg": np.random.rand(100, 12),
    "labels": ["AF"]
}

patient_200 = {
    "ecg": np.random.rand(200, 12),
    "labels": ["AF"]
}

samples_50 = task(patient_50)
samples_100 = task(patient_100)
samples_200 = task(patient_200)
print(samples_50, samples_100, samples_200)
```

#Task Configuration (Label Set) - comparing tasks with different number of labels (2 vs 4) at the same signal length of 100
```python
from pyhealth.tasks.ecg_classification import ECGMultiLabelTask
import numpy as np

task_simple = ECGMultiLabelTask(labels=["AF", "LBBB"])
task_complex = ECGMultiLabelTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

patient = {
    "ecg": np.random.rand(100, 12),
    "labels": ["AF"]
}

samples_simple = task_simple(patient)
samples_complex = task_complex(patient)
print(samples_simple, samples_complex)
```

