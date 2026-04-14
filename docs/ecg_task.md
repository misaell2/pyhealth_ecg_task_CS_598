# ECG Multi-Label Task Documentation

## Description
This task converts ECG signals into multi-label classification samples.

## Input Format
```python
{
  "ecg": np.ndarray (T, 12),
  "labels": List[str]
}