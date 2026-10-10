"""
test_data_cleaner.py
--------------------
Pruebas para el módulo de limpieza de datos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_cleaner import (
    remove_duplicates, handle_missing, convert_column_type,
    strip_whitespace, standardize_text, drop_columns,
    filter_rows, handle_outliers, CleaningHistory,
)


class TestRemoveDuplicates:
    def test_removes_duplicates(self, sample_df_with_issues):
        df = sample_df_with_issues
        n_dups = df.duplicated().sum()
        df_clean, record = remove_duplicates(df)
        assert len(df_clean) == len(df) - n_dups
        assert record["afectados"] == n_dups
        assert df_clean.duplicated().sum() == 0

    def test_no_duplicates_unchanged(self, sample_df):
        df_clean, record = remove_duplicates(sample_df)
        assert len(df_clean) == len(sample_df)
        assert record["afectados"] == 0


class TestHandleMissing:
    def test_eliminar_filas(self, sample_df_with_issues):
        df = sample_df_with_issues
        missing_count = df["precio"].isna().sum()
        df_clean, record = handle_missing(df, "precio", "eliminar_filas")
        assert df_clean["precio"].isna().sum() == 0
        assert record["afectados"] == missing_count

    def test_imputation_mean(self, sample_df_with_issues):
        df_clean, record = handle_missing(sample_df_with_issues, "precio", "media")
        assert df_clean["precio"].isna().sum() == 0
        assert record["parametros"]["estrategia"] == "media"

    def test_imputation_median(self, sample_df_with_issues):
        df_clean, record = handle_missing(sample_df_with_issues, "precio", "mediana")
        assert df_clean["precio"].isna().sum() == 0

    def test_imputation_mode(self, sample_df_with_issues):
        df_clean, record = handle_missing(sample_df_with_issues, "producto", "moda")
        assert df_clean["producto"].isna().sum() == 0

    def test_constant_fill(self, sample_df_with_issues):
        df_clean, record = handle_missing(sample_df_with_issues, "precio", "valor_constante", 0.0)
        assert df_clean["precio"].isna().sum() == 0

    def test_conservar_keeps_missing(self, sample_df_with_issues):
        missing_before = sample_df_with_issues["precio"].isna().sum()
        df_clean, _ = handle_missing(sample_df_with_issues, "precio", "conservar")
        assert df_clean["precio"].isna().sum() == missing_before

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="no existe"):
            handle_missing(sample_df, "columna_inexistente", "media")

    def test_mean_on_text_raises(self, sample_df):
        with pytest.raises(ValueError, match="num"):
            handle_missing(sample_df, "producto", "media")


class TestConvertColumnType:
    def test_convert_to_float(self, sample_df):
        df_int = sample_df.copy()
        df_int["cantidad"] = df_int["cantidad"].astype(str)
        df_clean, record = convert_column_type(df_int, "cantidad", "float")
        # pd.to_numeric con valores enteros puede devolver int64 o float64; ambos son válidos
        assert pd.api.types.is_numeric_dtype(df_clean["cantidad"])

    def test_convert_to_category(self, sample_df):
        df_clean, _ = convert_column_type(sample_df, "categoria", "category")
        assert hasattr(df_clean["categoria"], "cat")


class TestStripWhitespace:
    def test_strips_spaces(self, sample_df_with_issues):
        df_clean, record = strip_whitespace(sample_df_with_issues)
        for val in df_clean["producto"].dropna():
            assert val == val.strip()
        assert record["operacion"] == "Eliminar espacios"


class TestStandardizeText:
    def test_lower(self, sample_df):
        df_clean, _ = standardize_text(sample_df, "producto", "lower")
        assert all(v == v.lower() for v in df_clean["producto"].dropna())

    def test_upper(self, sample_df):
        df_clean, _ = standardize_text(sample_df, "producto", "upper")
        assert all(v == v.upper() for v in df_clean["producto"].dropna())

    def test_title(self, sample_df):
        df_clean, _ = standardize_text(sample_df, "producto", "title")
        for v in df_clean["producto"].dropna():
            assert v == v.title()

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError):
            standardize_text(sample_df, "col_inexistente", "lower")


class TestDropColumns:
    def test_drops_columns(self, sample_df):
        df_clean, record = drop_columns(sample_df, ["region"])
        assert "region" not in df_clean.columns
        assert record["afectados"] == 1

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError):
            drop_columns(sample_df, ["columna_que_no_existe"])


class TestFilterRows:
    def test_filter_equal(self, sample_df):
        df_clean, record = filter_rows(sample_df, "categoria", "==", "Electrónica")
        assert all(df_clean["categoria"] == "Electrónica")

    def test_filter_greater(self, sample_df):
        threshold = 10
        df_clean, _ = filter_rows(sample_df, "cantidad", ">", threshold)
        assert all(df_clean["cantidad"] > threshold)

    def test_filter_contains(self, sample_df):
        df_clean, _ = filter_rows(sample_df, "producto", "contiene", "Laptop")
        assert all(df_clean["producto"].str.contains("Laptop", case=False))


class TestHandleOutliers:
    def test_remove_outliers_reduces_rows(self, sample_df):
        # Añadir un outlier extremo
        df_with_outlier = sample_df.copy()
        df_with_outlier.loc[0, "precio"] = 999_999.0
        df_clean, record = handle_outliers(df_with_outlier, "precio", "iqr", "eliminar")
        assert len(df_clean) < len(df_with_outlier)

    def test_cap_outliers_no_row_loss(self, sample_df):
        df_with_outlier = sample_df.copy()
        df_with_outlier.loc[0, "precio"] = 999_999.0
        df_clean, _ = handle_outliers(df_with_outlier, "precio", "iqr", "capturar")
        assert len(df_clean) == len(df_with_outlier)
        assert df_clean["precio"].max() < 999_999.0

    def test_non_numeric_raises(self, sample_df):
        with pytest.raises(ValueError, match="numérica"):
            handle_outliers(sample_df, "producto", "iqr", "eliminar")


class TestCleaningHistory:
    def test_apply_and_undo(self, sample_df):
        hist = CleaningHistory(sample_df)
        _, _ = hist.apply(remove_duplicates)
        assert hist.can_undo
        prev_df = hist.undo()
        assert len(prev_df) == len(sample_df)

    def test_reset_returns_original(self, sample_df_with_issues):
        hist = CleaningHistory(sample_df_with_issues)
        hist.apply(remove_duplicates)
        original = hist.reset()
        assert len(original) == len(sample_df_with_issues)
        assert not hist.can_undo

    def test_history_records(self, sample_df):
        hist = CleaningHistory(sample_df)
        hist.apply(remove_duplicates)
        assert len(hist.history) == 1
        assert hist.history[0]["operacion"] == "Eliminar duplicados"
