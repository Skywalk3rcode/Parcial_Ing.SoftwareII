"""
test_data_profiler.py
---------------------
Pruebas para el módulo de perfilado y calidad de datos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_profiler import (
    profile_dataset,
    quick_quality_summary,
    _looks_like_date,
    _looks_like_numeric,
    _detect_outliers_iqr,
)


class TestLooksLikeDate:
    def test_detects_date_strings(self):
        series = pd.Series(["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"])
        assert _looks_like_date(series)

    def test_rejects_non_dates(self):
        series = pd.Series(["Laptop", "Tablet", "Teléfono", "Monitor"])
        assert not _looks_like_date(series)

    def test_empty_series_returns_false(self):
        assert not _looks_like_date(pd.Series([], dtype=str))


class TestLooksLikeNumeric:
    def test_detects_numeric_strings(self):
        series = pd.Series(["1.5", "2.0", "3.14", "100", "0.5"])
        assert _looks_like_numeric(series)

    def test_rejects_text(self):
        series = pd.Series(["abc", "def", "ghi", "jkl"])
        assert not _looks_like_numeric(series)


class TestDetectOutliersIqr:
    def test_detects_extreme_value(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 9999.0])
        outliers = _detect_outliers_iqr(series)
        assert 9999.0 in outliers.values

    def test_no_outliers_in_normal_distribution(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(100, 5, 200))
        outliers = _detect_outliers_iqr(series)
        # Puede haber algunos, pero no muchos
        assert len(outliers) < 20


class TestProfileDataset:
    def test_returns_expected_keys(self, sample_df):
        profile = profile_dataset(sample_df)
        required_keys = [
            "missing", "duplicates", "unique_values", "possible_id_columns",
            "dtypes", "dtype_suggestions", "numeric_as_text", "dates_as_text",
            "outliers", "format_issues", "quality_table", "quality_score",
        ]
        for key in required_keys:
            assert key in profile, f"Falta la clave: {key}"

    def test_quality_score_range(self, sample_df):
        profile = profile_dataset(sample_df)
        assert 0 <= profile["quality_score"] <= 100

    def test_detects_missing_values(self, sample_df_with_issues):
        profile = profile_dataset(sample_df_with_issues)
        missing_df = profile["missing"]
        cols_with_missing = missing_df[missing_df["Faltantes"] > 0]["Columna"].tolist()
        assert "precio" in cols_with_missing

    def test_detects_duplicates(self, sample_df_with_issues):
        profile = profile_dataset(sample_df_with_issues)
        assert profile["duplicates"]["cantidad"] > 0

    def test_no_false_positives_in_clean_df(self, sample_df):
        profile = profile_dataset(sample_df)
        # Un dataset limpio no debe tener duplicados
        assert profile["duplicates"]["cantidad"] == 0

    def test_quality_table_has_all_columns(self, sample_df):
        profile = profile_dataset(sample_df)
        qt = profile["quality_table"]
        assert len(qt) == sample_df.shape[1]
        assert "Columna" in qt.columns
        assert "Faltantes (%)" in qt.columns

    def test_detects_numeric_as_text(self):
        df = pd.DataFrame({
            "nombre": ["Ana", "Bob", "Carlos"],
            "precio_str": ["100.5", "200.0", "300.75"],
        })
        profile = profile_dataset(df)
        assert "precio_str" in profile["numeric_as_text"]

    def test_outlier_detection(self):
        df = pd.DataFrame({
            "valor": [10.0, 11.0, 12.0, 10.5, 11.5, 9999.0, 10.2, 11.3, 10.8, 11.1],
        })
        profile = profile_dataset(df)
        assert "valor" in profile["outliers"]


class TestQuickQualitySummary:
    def test_returns_expected_keys(self, sample_df):
        summary = quick_quality_summary(sample_df)
        keys = [
            "filas", "columnas", "valores_faltantes",
            "porcentaje_faltantes", "filas_duplicadas",
            "columnas_numericas", "columnas_categoricas",
        ]
        for key in keys:
            assert key in summary

    def test_correct_row_count(self, sample_df):
        summary = quick_quality_summary(sample_df)
        assert summary["filas"] == len(sample_df)

    def test_missing_count_with_issues(self, sample_df_with_issues):
        summary = quick_quality_summary(sample_df_with_issues)
        assert summary["valores_faltantes"] > 0
        assert summary["filas_duplicadas"] > 0
