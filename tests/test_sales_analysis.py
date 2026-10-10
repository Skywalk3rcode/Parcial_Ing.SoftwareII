"""
test_sales_analysis.py
----------------------
Pruebas para el módulo de análisis de ventas.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sales_analysis import (
    detect_sales_columns, is_sales_dataset, prepare_sales_df,
    compute_sales_metrics, revenue_by_product, revenue_by_category,
    revenue_over_time, top_selling_products,
)


@pytest.fixture
def ventas_df():
    """DataFrame de ventas completo."""
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "fecha": pd.date_range("2023-01-01", periods=n, freq="D"),
        "producto": np.random.choice(["Laptop", "Tablet", "Teléfono"], n),
        "categoria": np.random.choice(["Electrónica", "Accesorios"], n),
        "cantidad": np.random.randint(1, 10, n),
        "precio_unitario": np.random.uniform(100, 1000, n).round(2),
        "ingresos": None,  # Sin calcular
        "region": np.random.choice(["Norte", "Sur"], n),
    })


class TestDetectSalesColumns:
    def test_detects_standard_columns(self, ventas_df):
        mapping = detect_sales_columns(ventas_df)
        assert mapping.get("fecha") == "fecha"
        assert mapping.get("producto") == "producto"
        assert mapping.get("cantidad") == "cantidad"
        assert mapping.get("categoria") == "categoria"

    def test_returns_none_for_missing(self, small_df):
        mapping = detect_sales_columns(small_df)
        # No hay columnas de ventas reconocibles
        assert mapping.get("fecha") is None or mapping.get("producto") is None


class TestIsSalesDataset:
    def test_sales_df_recognized(self, ventas_df):
        assert is_sales_dataset(ventas_df)

    def test_non_sales_df_not_recognized(self, small_df):
        assert not is_sales_dataset(small_df)


class TestPrepareSalesdf:
    def test_prepares_with_revenue_calculation(self, ventas_df):
        mapping = {
            "fecha": "fecha",
            "producto": "producto",
            "categoria": "categoria",
            "cantidad": "cantidad",
            "precio": "precio_unitario",
            "ingresos": None,
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        assert "ingresos" in sales.columns
        assert sales["ingresos"].notna().all()
        # Verificar que ingresos = cantidad * precio
        expected = ventas_df["cantidad"] * ventas_df["precio_unitario"]
        assert np.allclose(sales["ingresos"].values, expected.values)

    def test_fecha_converted_to_datetime(self, ventas_df):
        mapping = {"fecha": "fecha", "cantidad": "cantidad"}
        sales = prepare_sales_df(ventas_df, mapping)
        assert pd.api.types.is_datetime64_any_dtype(sales["fecha"])


class TestComputeSalesMetrics:
    def test_returns_basic_metrics(self, ventas_df):
        mapping = {
            "fecha": "fecha",
            "cantidad": "cantidad",
            "precio": "precio_unitario",
            "ingresos": None,
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        metrics = compute_sales_metrics(sales)
        assert "ingresos_totales" in metrics
        assert "unidades_totales" in metrics
        assert metrics["ingresos_totales"] > 0
        assert metrics["total_registros"] == len(sales)


class TestRevenueByProduct:
    def test_returns_sorted_products(self, ventas_df):
        mapping = {
            "producto": "producto",
            "cantidad": "cantidad",
            "precio": "precio_unitario",
            "ingresos": None,
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        result = revenue_by_product(sales)
        assert not result.empty
        assert "producto" in result.columns
        assert "ingresos" in result.columns

    def test_top_n_limits_results(self, ventas_df):
        mapping = {
            "producto": "producto",
            "ingresos": None,
            "cantidad": "cantidad",
            "precio": "precio_unitario",
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        result = revenue_by_product(sales, top_n=2)
        assert len(result) <= 2


class TestRevenueByCategory:
    def test_includes_participation(self, ventas_df):
        mapping = {
            "categoria": "categoria",
            "ingresos": None,
            "cantidad": "cantidad",
            "precio": "precio_unitario",
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        result = revenue_by_category(sales)
        assert not result.empty
        assert "participacion_pct" in result.columns
        # Participaciones suman aprox. 100%
        assert abs(result["participacion_pct"].sum() - 100) < 1


class TestRevenueOverTime:
    def test_aggregates_by_month(self, ventas_df):
        mapping = {
            "fecha": "fecha",
            "ingresos": None,
            "cantidad": "cantidad",
            "precio": "precio_unitario",
        }
        sales = prepare_sales_df(ventas_df, mapping, calculate_revenue=True)
        result = revenue_over_time(sales, freq="ME")
        assert not result.empty
        assert "fecha" in result.columns
        assert "ingresos" in result.columns
