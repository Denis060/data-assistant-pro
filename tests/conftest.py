"""Test fixtures and utilities for Data Assistant Pro tests."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_clean_data() -> pd.DataFrame:
    """Create a clean sample dataset for testing."""
    return pd.DataFrame(
        {
            "id": range(1, 11),
            "name": [f"Person_{i}" for i in range(1, 11)],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "salary": [
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
                120000,
                130000,
                140000,
            ],
            "department": [
                "IT",
                "HR",
                "IT",
                "Finance",
                "HR",
                "IT",
                "Finance",
                "IT",
                "HR",
                "Finance",
            ],
            "score": [8.5, 7.2, 9.1, 6.8, 8.9, 7.5, 9.3, 8.1, 7.8, 9.0],
        }
    )


@pytest.fixture
def sample_dirty_data() -> pd.DataFrame:
    """Create a dirty sample dataset with missing values, duplicates, and outliers."""
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],  # Duplicate ID
        "name": [
            "Alice",
            "Bob",
            None,
            "David",
            "Eva",
            "Frank",
            "Grace",
            "Henry",
            "Ivy",
            "Jack",
            "Jack",
        ],  # Missing value, duplicate
        "age": [
            25,
            30,
            35,
            40,
            np.nan,
            50,
            55,
            200,
            65,
            70,
            70,
        ],  # Missing value, outlier
        "salary": [
            50000,
            60000,
            70000,
            80000,
            90000,
            None,
            110000,
            120000,
            130000,
            140000,
            140000,
        ],  # Missing value
        "department": [
            "IT",
            "HR",
            "IT",
            "Finance",
            "HR",
            "IT",
            "Finance",
            "IT",
            "HR",
            "Finance",
            "Finance",
        ],
        "score": [8.5, 7.2, 9.1, 6.8, 8.9, 7.5, 9.3, 8.1, 7.8, 9.0, 9.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_numerical_data() -> pd.DataFrame:
    """Create a numerical dataset for outlier detection testing."""
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 100)
    outliers = [150, -50, 200]  # Clear outliers
    all_data = np.concatenate([normal_data, outliers])

    return pd.DataFrame(
        {"values": all_data, "category": ["A"] * 50 + ["B"] * 50 + ["C"] * 3}
    )


@pytest.fixture
def sample_modeling_data() -> pd.DataFrame:
    """Create a dataset suitable for machine learning testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_samples),
            "feature2": np.random.normal(5, 2, n_samples),
            "feature3": np.random.randint(0, 10, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "target_continuous": np.random.normal(10, 3, n_samples),
            "target_binary": np.random.choice([0, 1], n_samples),
            "target_multiclass": np.random.choice([0, 1, 2], n_samples),
        }
    )


@pytest.fixture
def temp_csv_file(tmp_path, sample_clean_data):
    """Create a temporary CSV file for testing file operations."""
    csv_file = tmp_path / "test_data.csv"
    sample_clean_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def outlier_detection_params() -> Dict[str, Any]:
    """Standard parameters for outlier detection testing."""
    return {
        "iqr_multiplier": 1.5,
        "zscore_threshold": 3.0,
        "modified_zscore_threshold": 3.5,
    }
