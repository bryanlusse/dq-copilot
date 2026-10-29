import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10.0, 20.0, None, 40.0, 50.0],
            "category": ["a", "a", "b", "b", "a"],
        }
    )


@pytest.fixture
def edge_df() -> pd.DataFrame:
    """One column per extra: empty strings, negatives/zeros, future dates."""
    return pd.DataFrame(
        {
            "name": ["a", "", " ", None, "a"],  # empty_count 2 ('' and ' '), null_count 1
            "delta": [-1.0, 0.0, 0.0, 5.0, None],  # negative 1, zero 2, null 1
            "when": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2200-01-01", None, "2020-01-01"]
            ),  # future_count 1, null 1
        }
    )


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """500 clean taxi-like trips: deterministic, no downloads."""
    rng = np.random.default_rng(42)
    n = 500
    pickup = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 30 * 24 * 3600, n), unit="s"
    )
    distance = rng.gamma(2.0, 2.0, n).clip(0.1, 40).round(2)
    return pd.DataFrame(
        {
            "trip_id": np.arange(n),
            "pickup_at": pickup,
            "dropoff_at": pickup + pd.to_timedelta(rng.integers(3, 60, n), unit="m"),
            "passenger_count": rng.integers(1, 5, n),
            "trip_distance": distance,
            "fare_amount": (3 + 2.5 * distance).round(2),
            "payment_type": rng.choice(["card", "cash", "app"], n, p=[0.6, 0.25, 0.15]),
            "vendor_id": rng.choice(["V1", "V2"], n),
        }
    )


def inject_nulls(df: pd.DataFrame, col: str, frac: float = 0.1, seed: int = 0):
    """Returns (corrupted_df, ground_truth) where ground_truth lists affected rows."""
    out = df.copy()
    idx = out.sample(frac=frac, random_state=seed).index
    out.loc[idx, col] = None
    return out, {"defect": "nulls", "column": col, "rows": sorted(idx)}


def inject_outliers(df: pd.DataFrame, col: str, factor: float = 100, n: int = 5, seed: int = 0):
    out = df.copy()
    idx = out.sample(n=n, random_state=seed).index
    out.loc[idx, col] = out.loc[idx, col] * factor
    return out, {"defect": "outliers", "column": col, "rows": sorted(idx)}


def inject_duplicates(df: pd.DataFrame, key: str, n: int = 5, seed: int = 0):
    dup = df.sample(n=n, random_state=seed)
    out = pd.concat([df, dup], ignore_index=True)
    return out, {"defect": "duplicates", "column": key, "rows": sorted(dup.index)}


def inject_time_reversal(df: pd.DataFrame, n: int = 5, seed: int = 0):
    """dropoff_at before pickup_at: a cross-column rule a profiler alone won't suggest."""
    out = df.copy()
    idx = out.sample(n=n, random_state=seed).index
    out.loc[idx, "dropoff_at"] = out.loc[idx, "pickup_at"] - pd.Timedelta(minutes=5)
    return out, {"defect": "time_reversal", "column": "dropoff_at", "rows": sorted(idx)}
