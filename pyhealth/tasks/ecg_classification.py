from pyhealth.tasks.ecg_classification import ECGMultiLabelTask
import numpy as np

task = ECGMultiLabelTask(labels=["AF", "I-AVB", "LBBB", "RBBB"])

patient = {
    "ecg": np.random.rand(100, 12),
    "labels": ["AF", "RBBB"]
}

samples = task(patient)
print(samples)