"""
Data Acquisition Module
------------------------
Per patent spec, section "Data Acquisition Module": collects historical
fabrication sensor data. This implementation uses the publicly available
SECOM semiconductor manufacturing dataset (the preferred embodiment named
explicitly in the patent), containing process sensor measurements and
wafer pass/fail labels.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw_data(data_path: Path = DATA_DIR / "secom.data",
                   labels_path: Path = DATA_DIR / "secom_labels.data") -> pd.DataFrame:
    """Load raw SECOM sensor readings and pass/fail labels into one DataFrame.

    secom.data: whitespace-separated sensor readings, 1567 rows x 590 sensors.
    secom_labels.data: "<label> <timestamp>" per row. label -1 = pass, 1 = fail.
    """
    sensors = pd.read_csv(data_path, sep=r"\s+", header=None)
    sensors.columns = [f"sensor_{i+1}" for i in range(sensors.shape[1])]

    # The regex whitespace separator does not respect the quoted "date time" field
    # (quoting is only honored by pandas' C engine with a single-char delimiter),
    # so parse each line manually instead of relying on read_csv to split it.
    labels, timestamps = [], []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, rest = line.split(" ", 1)
            ts_str = rest.strip().strip('"')
            labels.append(int(label_str))
            timestamps.append(ts_str)
    labels_raw = pd.DataFrame({"label": labels, "timestamp": pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S")})

    # Map -1 (pass) -> 0, 1 (fail) -> 1  (1 = failure, matches "failure probability" in spec)
    df = sensors.copy()
    df["timestamp"] = labels_raw["timestamp"]
    df["label"] = labels_raw["label"].map({-1: 0, 1: 1})
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {df.shape[0]} wafer lots, {df.shape[1]-2} sensor readings each.")
    print(f"Failure rate: {df['label'].mean():.2%}")
