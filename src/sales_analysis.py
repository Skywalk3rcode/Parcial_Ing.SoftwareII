"""
sales_analysis.py
-----------------
Módulo de análisis especializado para conjuntos de datos de ventas.

Detecta columnas compatibles, calcula métricas clave y genera
los datos necesarios para visualizaciones de ventas.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# Detección de columnas de ventas
# ─────────────────────────────────────────────

# Palabras clave para autodetección de columnas
_KEYWORDS = {
    "fecha": ["fecha", "date", "time", "día", "dia", "period", "periodo", "mes", "año"],
    "producto": ["producto", "product", "item", "articulo", "artículo", "nombre"],
    "categoria": ["categoria", "categoría", "category", "tipo", "type", "grupo", "group"],
    "cantidad": ["cantidad", "quantity", "qty", "units", "unidades", "count"],
    "precio": ["precio", "price", "costo", "cost", "valor_unitario", "unit_price"],
    "ingresos": ["ingreso", "revenue", "total", "importe", "monto", "amount", "venta", "sale"],
    "cliente": ["cliente", "customer", "client", "comprador"],
    "region": ["region", "región", "zona", "area", "ciudad", "ciudad", "pais", "país"],
    "vendedor": ["vendedor", "seller", "agente", "rep", "representante"],
}


def detect_sales_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """
    Intenta detectar automáticamente las columnas de ventas mediante palabras clave.

    Returns
    -------
    mapping : dict con {rol: nombre_columna o None}.
    """
    cols_lower = {col.lower(): col for col in df.columns}
    mapping: dict[str, Optional[str]] = {role: None for role in _KEYWORDS}

    for role, keywords in _KEYWORDS.items():
        for kw in keywords:
            for col_lower, col_orig in cols_lower.items():
                if kw in col_lower:
                    # Validar compatibilidad de tipo
                    if role in ("fecha",) and not _is_date_like(df[col_orig]):
                        continue
                    if role in ("cantidad", "precio", "ingresos") and not pd.api.types.is_numeric_dtype(df[col_orig]):
                        continue
                    mapping[role] = col_orig
                    break
            if mapping[role]:
                break

    return mapping


def _is_date_like(series: pd.Series) -> bool:
    """Comprueba si una columna puede ser fecha."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        converted = pd.to_datetime(series.dropna().head(50), errors="coerce")
        return converted.notna().mean() > 0.7
    except Exception:
        return False


def is_sales_dataset(df: pd.DataFrame, mapping: Optional[dict] = None) -> bool:
    """
    Determina si el DataFrame es compatible con un análisis de ventas.
    Requiere al menos: (fecha o producto) + (cantidad o ingresos).
    """
    if mapping is None:
        mapping = detect_sales_columns(df)
    has_time_or_product = mapping.get("fecha") or mapping.get("producto")
    has_values = mapping.get("cantidad") or mapping.get("ingresos")
    return bool(has_time_or_product and has_values)


# ─────────────────────────────────────────────
# Preparación del DataFrame de ventas
# ─────────────────────────────────────────────

def prepare_sales_df(
    df: pd.DataFrame,
    mapping: dict[str, Optional[str]],
    calculate_revenue: bool = False,
) -> pd.DataFrame:
    """
    Crea un DataFrame de ventas normalizado a partir del mapping de columnas.

    Parameters
    ----------
    mapping          : dict {rol: nombre_columna}.
    calculate_revenue: Si no hay columna de ingresos, calcularla como cantidad * precio.
    """
    sales = pd.DataFrame()

    for role, col in mapping.items():
        if col and col in df.columns:
            sales[role] = df[col].copy()

    # Convertir fecha
    if "fecha" in sales.columns:
        sales["fecha"] = pd.to_datetime(sales["fecha"], errors="coerce")
        sales = sales.dropna(subset=["fecha"])

    # Calcular ingresos si no existen
    if "ingresos" not in sales.columns or sales["ingresos"].isna().all():
        if calculate_revenue and "cantidad" in sales.columns and "precio" in sales.columns:
            sales["ingresos"] = sales["cantidad"] * sales["precio"]

    return sales.reset_index(drop=True)


# ─────────────────────────────────────────────
# Métricas de ventas
# ─────────────────────────────────────────────

def compute_sales_metrics(sales: pd.DataFrame) -> dict[str, Any]:
    """
    Calcula métricas clave de ventas a partir del DataFrame normalizado.
    """
    metrics: dict[str, Any] = {}

    if "ingresos" in sales.columns:
        metrics["ingresos_totales"] = float(sales["ingresos"].sum())
        metrics["ingreso_promedio"] = float(sales["ingresos"].mean())
        metrics["ingreso_mediano"] = float(sales["ingresos"].median())

    if "cantidad" in sales.columns:
        metrics["unidades_totales"] = float(sales["cantidad"].sum())
        metrics["unidades_promedio"] = float(sales["cantidad"].mean())

    if "precio" in sales.columns:
        metrics["precio_promedio"] = float(sales["precio"].mean())

    metrics["total_registros"] = len(sales)

    if "fecha" in sales.columns:
        valid_dates = sales["fecha"].dropna()
        if len(valid_dates) > 0:
            metrics["fecha_inicio"] = valid_dates.min().strftime("%Y-%m-%d")
            metrics["fecha_fin"] = valid_dates.max().strftime("%Y-%m-%d")
            metrics["dias_activos"] = int((valid_dates.max() - valid_dates.min()).days + 1)

    return metrics


def revenue_by_product(sales: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Ingresos y unidades agrupados por producto."""
    if "producto" not in sales.columns:
        return pd.DataFrame()

    agg: dict[str, Any] = {}
    if "ingresos" in sales.columns:
        agg["ingresos"] = "sum"
    if "cantidad" in sales.columns:
        agg["cantidad"] = "sum"

    if not agg:
        return pd.DataFrame()

    result = (
        sales.groupby("producto", observed=True)
        .agg(agg)
        .reset_index()
        .sort_values("ingresos" if "ingresos" in agg else "cantidad", ascending=False)
        .head(top_n)
    )
    return result


def revenue_by_category(sales: pd.DataFrame) -> pd.DataFrame:
    """Ingresos y participación por categoría."""
    if "categoria" not in sales.columns:
        return pd.DataFrame()

    agg: dict[str, Any] = {}
    if "ingresos" in sales.columns:
        agg["ingresos"] = "sum"
    if "cantidad" in sales.columns:
        agg["cantidad"] = "sum"

    if not agg:
        return pd.DataFrame()

    result = (
        sales.groupby("categoria", observed=True)
        .agg(agg)
        .reset_index()
        .sort_values("ingresos" if "ingresos" in agg else "cantidad", ascending=False)
    )

    if "ingresos" in result.columns:
        total = result["ingresos"].sum()
        result["participacion_pct"] = (result["ingresos"] / total * 100).round(2)

    return result


def revenue_over_time(
    sales: pd.DataFrame,
    freq: str = "ME",
    value_col: str = "ingresos",
) -> pd.DataFrame:
    """Evolución temporal de ingresos o unidades."""
    if "fecha" not in sales.columns or value_col not in sales.columns:
        return pd.DataFrame()

    temp = sales[["fecha", value_col]].copy()
    temp["fecha"] = pd.to_datetime(temp["fecha"], errors="coerce")
    temp = temp.dropna().set_index("fecha").sort_index()
    result = temp[value_col].resample(freq).sum().reset_index()
    result.columns = ["fecha", value_col]
    return result


def top_selling_products(sales: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Productos más vendidos por unidades."""
    if "producto" not in sales.columns or "cantidad" not in sales.columns:
        return pd.DataFrame()

    result = (
        sales.groupby("producto", observed=True)["cantidad"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    result.columns = ["Producto", "Unidades vendidas"]
    return result


def period_comparison(
    sales: pd.DataFrame,
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str,
    value_col: str = "ingresos",
) -> dict[str, Any]:
    """Compara métricas entre dos períodos."""
    if "fecha" not in sales.columns or value_col not in sales.columns:
        return {}

    p1 = sales[
        (sales["fecha"] >= period1_start) & (sales["fecha"] <= period1_end)
    ][value_col].sum()

    p2 = sales[
        (sales["fecha"] >= period2_start) & (sales["fecha"] <= period2_end)
    ][value_col].sum()

    change = p2 - p1
    change_pct = (change / p1 * 100) if p1 != 0 else 0.0

    return {
        "periodo_1": {"inicio": period1_start, "fin": period1_end, "total": float(p1)},
        "periodo_2": {"inicio": period2_start, "fin": period2_end, "total": float(p2)},
        "variacion": float(change),
        "variacion_pct": round(float(change_pct), 2),
    }


def sales_by_region(sales: pd.DataFrame) -> pd.DataFrame:
    """Ingresos agrupados por región."""
    if "region" not in sales.columns:
        return pd.DataFrame()

    agg: dict[str, Any] = {}
    if "ingresos" in sales.columns:
        agg["ingresos"] = "sum"
    if "cantidad" in sales.columns:
        agg["cantidad"] = "sum"

    return (
        sales.groupby("region", observed=True)
        .agg(agg)
        .reset_index()
        .sort_values("ingresos" if "ingresos" in agg else "cantidad", ascending=False)
    )


def activity_by_day(sales: pd.DataFrame, value_col: str = "ingresos") -> pd.DataFrame:
    """Actividad por día de la semana."""
    if "fecha" not in sales.columns or value_col not in sales.columns:
        return pd.DataFrame()

    temp = sales[["fecha", value_col]].copy()
    temp["fecha"] = pd.to_datetime(temp["fecha"], errors="coerce")
    temp = temp.dropna()
    temp["dia_semana"] = temp["fecha"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_order_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    day_map = dict(zip(day_order, day_order_es))
    temp["dia_semana"] = temp["dia_semana"].map(day_map)

    result = (
        temp.groupby("dia_semana", observed=True)[value_col]
        .sum()
        .reindex(day_order_es)
        .reset_index()
    )
    result.columns = ["Día", value_col]
    return result
