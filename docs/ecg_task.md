# ECG Multi-Label Task Documentation

## Overview
The `ECGMultiLabelTask` is a standalone task implementation for converting
raw ECG signals into model-ready samples for **multi-label classification**.

This task is inspired by *Nonaka & Seita (2021)*, which studies deep learning
approaches for ECG diagnosis across multiple datasets.

---

## Task Description
This task processes **12-lead ECG signals** and maps them to a set of
diagnostic labels. Each patient may have **multiple simultaneous conditions**,
making this a multi-label classification problem.

### Supported Labels (example)
- Atrial Fibrillation (**AF**)
- First-degree Atrioventricular Block (**I-AVB**)
- Left Bundle Branch Block (**LBBB**)
- Right Bundle Branch Block (**RBBB**)

---

## Input Format

Each patient record must be a dictionary with the following structure:

```python
{
  "ecg": np.ndarray of shape (T, 12),  # ECG signal (T timesteps, 12 leads)
  "labels": List[str]                  # List of diagnostic labels
}