import os
import sys
import random
import importlib.util
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prevent the local repo's ./pyhealth folder from shadowing installed PyHealth
sys.path = [p for p in sys.path if Path(p).resolve() != PROJECT_ROOT]

from pyhealth.datasets import SampleSignalDataset

# Load the local standalone task file directly from disk
task_file = PROJECT_ROOT / "pyhealth" / "tasks" / "ecg_classification.py"
spec = importlib.util.spec_from_file_location("local_ecg_classification", task_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ECGMultiLabelCardiologyTask = module.ECGMultiLabelCardiologyTask

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Put one or more dataset folders here.
DATASET_ROOTS = [
    PROJECT_ROOT / "datasets" / "root" / "ptb-xl",
    # PROJECT_ROOT / "datasets" / "root" / "georgia",
    # PROJECT_ROOT / "datasets" / "root" / "cpsc_2018",
]

# Use None for full dataset processing.
# Set to an integer only when debugging.
MAX_RECORDS_PER_DATASET = None

LABEL_SPACE = [
    "164889003",
    "164890007",
    "426627000",
    "284470004",
    "427172004",
]

EPOCH_SEC = 10
SHIFT_SEC = 5
SAMPLING_RATE = 500

# How many sample summaries to print per dataset
NUM_EXAMPLE_SAMPLES = 3


def collect_physionet_records(dataset_root: Path, max_records: int = None) -> List[Dict]:
    """Collect records from a PhysioNet Challenge-style dataset layout."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    visits: List[Dict] = []
    subdirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    for group_dir in subdirs:
        mat_files = sorted(group_dir.glob("*.mat"))

        for mat_path in mat_files:
            hea_path = mat_path.with_suffix(".hea")
            if not hea_path.exists():
                print(f"Skipping {mat_path}: missing matching .hea file")
                continue

            record_id = mat_path.stem

            visits.append(
                {
                    "load_from_path": str(group_dir),
                    "patient_id": f"{group_dir.name}_{record_id}",
                    "signal_file": mat_path.name,
                    "label_file": hea_path.name,
                }
            )

            if max_records is not None and len(visits) >= max_records:
                return visits

    return visits


def robust_parse_header_metadata(header_path: str) -> Dict[str, List[str]]:
    """Robustly parse #Dx, #Sex, and #Age from a PhysioNet-style .hea header."""
    dx_codes: List[str] = []
    sex: List[str] = []
    age: List[str] = []

    with open(header_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line.startswith("#"):
                continue

            key, sep, value = line[1:].partition(":")
            if not sep:
                continue

            key = key.strip().lower()
            value = value.strip()

            if key == "dx":
                dx_codes = [x.strip() for x in value.split(",") if x.strip()]
            elif key == "sex":
                sex = [value] if value else []
            elif key == "age":
                age = [value] if value else []

    return {
        "dx_codes": dx_codes,
        "sex": sex,
        "age": age,
    }


def preview_header(header_path: Path, num_lines: int = 20) -> None:
    """Print the first few lines of a header file for debugging."""
    print(f"\n--- Header preview: {header_path} ---")
    try:
        with open(header_path, "r", encoding="utf-8-sig", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(line.rstrip("\n"))
    except Exception as e:
        print(f"Could not read header {header_path}: {e}")
    print("--- End header preview ---\n")


def summarize_sample(sample: Dict) -> None:
    """Print a compact summary of one generated sample."""
    print("first sample keys:", list(sample.keys()))

    signal = sample.get("signal")
    label = sample.get("label")

    print("patient_id:", sample.get("patient_id"))
    print("visit_id:", sample.get("visit_id"))
    print("record_id:", sample.get("record_id"))
    print("Sex:", sample.get("Sex"))
    print("Age:", sample.get("Age"))

    if isinstance(signal, np.ndarray):
        print("signal shape:", signal.shape)
        print("signal dtype:", signal.dtype)
    else:
        print("signal type:", type(signal))

    if isinstance(label, np.ndarray):
        print("label shape:", label.shape)
        print("label dtype:", label.dtype)
        print("label values:", label.tolist())
    else:
        print("label type:", type(label))
        print("label value:", label)


def summarize_dataset_outputs(dataset_name: str, visits: List[Dict], samples: List[Dict]) -> None:
    """Print high-level summary statistics for a processed dataset."""
    print(f"\n=== Summary for {dataset_name} ===")
    print("num visits:", len(visits))
    print("num generated samples:", len(samples))

    if len(samples) == 0:
        print("No samples generated.")
        return

    num_positive = 0
    label_sums = np.zeros(len(LABEL_SPACE), dtype=np.int64)

    sex_counts = {}
    age_present = 0

    for sample in samples:
        label = sample.get("label")
        if isinstance(label, np.ndarray):
            if label.sum() > 0:
                num_positive += 1
            label_sums += label.astype(np.int64)

        sex_val = sample.get("Sex", [])
        if isinstance(sex_val, list) and len(sex_val) > 0:
            sex_key = sex_val[0]
            sex_counts[sex_key] = sex_counts.get(sex_key, 0) + 1

        age_val = sample.get("Age", [])
        if isinstance(age_val, list) and len(age_val) > 0:
            age_present += 1

    print("samples with >=1 positive label:", num_positive)
    print("positive-label rate:", f"{num_positive / len(samples):.4f}")
    print("label counts by configured label:")
    for code, count in zip(LABEL_SPACE, label_sums.tolist()):
        print(f"  {code}: {count}")

    print("sex counts:", sex_counts)
    print("samples with age present:", age_present)

    print(f"\nFirst {min(NUM_EXAMPLE_SAMPLES, len(samples))} sample summaries:")
    for i in range(min(NUM_EXAMPLE_SAMPLES, len(samples))):
        print(f"\n--- sample {i} ---")
        summarize_sample(samples[i])


def main():
    print("Project root:", PROJECT_ROOT)
    print("Task file:", task_file)
    print("Dataset roots:")
    for root in DATASET_ROOTS:
        print(" ", root)
    print("Max records per dataset:", MAX_RECORDS_PER_DATASET)

    task = ECGMultiLabelCardiologyTask(
        labels=LABEL_SPACE,
        epoch_sec=EPOCH_SEC,
        shift=SHIFT_SEC,
        sampling_rate=SAMPLING_RATE,
    )

    # Test-side monkeypatch so the task reads actual .hea metadata reliably.
    task._parse_header_metadata = types.MethodType(
        lambda self, header_path: robust_parse_header_metadata(header_path),
        task,
    )

    all_samples = []
    total_visits = 0

    for dataset_root in DATASET_ROOTS:
        print(f"\n==============================")
        print(f"Processing dataset: {dataset_root.name}")
        print(f"==============================")

        visits = collect_physionet_records(
            dataset_root,
            max_records=MAX_RECORDS_PER_DATASET,
        )

        print("num collected visits:", len(visits))
        if len(visits) == 0:
            print(f"No valid .mat/.hea pairs found under {dataset_root}")
            continue

        print("first visit:", visits[0])

        # Show one header preview only per dataset
        first_header_path = Path(visits[0]["load_from_path"]) / visits[0]["label_file"]
        preview_header(first_header_path, num_lines=20)

        parsed_header = robust_parse_header_metadata(str(first_header_path))
        print("directly parsed header metadata:", parsed_header)

        samples = task(visits)

        summarize_dataset_outputs(dataset_root.name, visits, samples)

        sample_dataset = SampleSignalDataset(
            samples=samples,
            dataset_name=dataset_root.name,
            task_name=getattr(task, "task_name", "ECGMultiLabelCardiologyTask"),
        )

        print("\nwrapped dataset type:", type(sample_dataset))
        print("wrapped sample count:", len(samples))
        print("wrapped dataset task_name:", getattr(sample_dataset, "task_name", "N/A"))

        if hasattr(sample_dataset, "input_info"):
            print("wrapped dataset input_info:", sample_dataset.input_info)

        if hasattr(sample_dataset, "stat"):
            print("wrapped dataset stat():")
            print(sample_dataset.stat())

        all_samples.extend(samples)
        total_visits += len(visits)

    print(f"\n========================================")
    print("Overall run summary")
    print("========================================")
    print("total visits across datasets:", total_visits)
    print("total samples across datasets:", len(all_samples))

    if len(all_samples) == 0:
        raise RuntimeError("No samples were generated from any dataset root.")


if __name__ == "__main__":
    main()