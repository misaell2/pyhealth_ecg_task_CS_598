ECG Multi-Label Cardiology Task
===============================

Overview
--------

This module provides an implementation of a multi-label ECG classification task
for the PyHealth framework.

The task processes 12-lead electrocardiogram (ECG) signals and predicts multiple
cardiac conditions simultaneously. It extends the PyHealth ``BaseTask`` class and
integrates with PyHealth datasets and models.

Key Features
------------

- Multi-label classification support
- PhysioNet-style ECG file handling (.mat and .hea)
- Sliding window segmentation
- Metadata extraction (age, sex, diagnosis codes)
- Compatibility with PyHealth pipelines

Input Format
------------

Each patient visit is represented as a dictionary:

::

    {
        "load_from_path": "...",
        "patient_id": "...",
        "signal_file": "record.mat",
        "label_file": "record.hea",
    }

Output Format
-------------

Each processed sample contains:

::

    {
        "signal": numpy.ndarray,
        "label": numpy.ndarray,
        "patient_id": str,
        "visit_id": str,
        "record_id": str,
        "Sex": str,
        "Age": int,
    }

Parameters
----------

- ``labels``: list of target diagnosis labels
- ``epoch_sec``: window size in seconds
- ``shift``: step size between windows
- ``sampling_rate``: signal sampling frequency

Ablation Support
----------------

This task enables experimentation across:

- Label set variation (task-level)
- Temporal segmentation (data-level)

Example
-------

::

    from pyhealth.tasks.ecg_classification import ECGMultiLabelCardiologyTask

    task = ECGMultiLabelCardiologyTask(
        labels=["AF", "RBBB"],
        epoch_sec=10,
        shift=5,
        sampling_rate=500,
    )

    samples = task(visit_dict)

Module Reference
----------------

.. automodule:: pyhealth.tasks.ecg_classification
    :members:
    :undoc-members:
    :show-inheritance:
