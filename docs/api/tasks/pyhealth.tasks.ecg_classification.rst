ECGMultiLabelTask
=================

ECGMultiLabelTask is a standalone task for converting ECG signals into
multi-label classification samples.

This task is inspired by Nonaka & Seita (2021), which benchmarks deep learning
architectures for ECG diagnosis. It supports flexible task configurations,
including varying input lengths and label sets for ablation studies.

Overview
--------

- Input: 12-lead ECG signal of shape (T, 12)
- Output: Multi-hot encoded label vector
- Supports multi-label classification
- Designed for synthetic data testing and fast execution

Module Reference
----------------

.. automodule:: pyhealth.tasks.ecg_classification
   :members:
   :undoc-members:
   :show-inheritance: