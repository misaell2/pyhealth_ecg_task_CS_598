# ECG Multi-Label Cardiology Task

## Overview

This module implements a PyHealth-compatible task for ECG multi-label classification.

The task processes 12-lead electrocardiogram (ECG) signals and predicts multiple cardiac conditions simultaneously. It extends PyHealth’s `BaseTask` interface and is designed to integrate seamlessly with PyHealth datasets and models.

---

## Key Features

- Supports **multi-label classification**
- Handles **PhysioNet-style ECG data** (`.mat` + `.hea`)
- Performs **sliding window segmentation**
- Extracts metadata (age, sex, diagnosis codes)
- Fully compatible with PyHealth pipelines

---

## Input Format

Each patient visit must be represented as:

```python
{
  "load_from_path": "...",
  "patient_id": "...",
  "signal_file": "record.mat",
  "label_file": "record.hea",
}
