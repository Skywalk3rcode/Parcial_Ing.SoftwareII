"""
conftest.py
-----------
Fixtures compartidas para las pruebas de pytest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de ventas de ejemplo con varios tipos de dato."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "fecha": pd.date_range("2023-01-01", periods=n, freq="D"),
        "producto": np.random.choice(["Laptop", "Tablet", "Teléfono", "Monitor"], n),
        "categoria": np.random.choice(["Electrónica", "Accesorios"], n),
        "cantidad": np.random.randint(1, 20, n),
        "precio": np.random.uniform(50, 1500, n).round(2),
        "region": np.random.choice(["Norte", "Sur", "Centro"], n),
    })


@pytest.fixture
def sample_df_with_issues(sample_df) -> pd.DataFrame:
    """DataFrame con valores faltantes, duplicados y tipos incorrectos."""
    df = sample_df.copy()
    # Valores faltantes
    df.loc[0:9, "precio"] = None
    df.loc[5:8, "producto"] = None
    # Duplicados
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Espacios en texto
    df.loc[10:12, "producto"] = "  Laptop  "
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    """DataFrame pequeño (5 filas) para pruebas que requieren datos insuficientes para ML."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 5.0, 4.0, 5.0],
        "cat": ["a", "b", "a", "b", "a"],
    })
