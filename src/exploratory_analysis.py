"""
exploratory_analysis.py
-----------------------
Módulo de Análisis Exploratorio de Datos (EDA).

Calcula estadísticas descriptivas para columnas numéricas, categóricas y temporales,
y construye los datos necesarios para generar gráficos interactivos.
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
# Estadísticas numéricas
# ─────────────────────────────────────────────

def numeric_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """
    Calcula estadísticas descriptivas completas para una columna numérica.

    Returns
    -------
    dict con media, mediana, moda, std, varianza, min, max, cuartiles, IQR, etc.
    """
    series = df[column].dropna()
    if len(series) == 0:
        return {"error": f"La columna '{column}' no tiene valores válidos."}

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))

    modes = series.mode()
    moda = float(modes.iloc[0]) if len(modes) > 0 else None
    es_multimodal = len(modes) > 1

    return {
        "columna": column,
        "count": int(series.count()),
        "nulos": int(df[column].isna().sum()),
        "media": float(series.mean()),
        "mediana": float(series.median()),
        "moda": moda,
        "es_multimodal": es_multimodal,
        "std": float(series.std()),
        "varianza": float(series.var()),
        "min": float(series.min()),
        "max": float(series.max()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p5": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
        "asimetria": float(series.skew()),
        "curtosis": float(series.kurtosis()),
    }


def all_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas para todas las columnas numéricas.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    rows = []
    for col in num_cols:
        s = numeric_stats(df, col)
        if "error" not in s:
            rows.append(s)
    return pd.DataFrame(rows).set_index("columna")


# ─────────────────────────────────────────────
# Estadísticas categóricas
# ─────────────────────────────────────────────

def categorical_stats(df: pd.DataFrame, column: str, top_n: int = 20) -> dict[str, Any]:
    """
    Calcula frecuencias y distribución de categorías para una columna de texto.
    """
    series = df[column].dropna()
    if len(series) == 0:
        return {"error": f"La columna '{column}' no tiene valores válidos."}

    freq = series.value_counts()
    freq_pct = series.value_counts(normalize=True) * 100

    return {
        "columna": column,
        "total": len(series),
        "nulos": int(df[column].isna().sum()),
        "unicos": int(series.nunique()),
        "top_categorias": freq.head(top_n).to_dict(),
        "top_porcentajes": freq_pct.head(top_n).round(2).to_dict(),
        "categoria_mas_frecuente": str(freq.index[0]) if len(freq) > 0 else None,
        "frecuencia_maxima": int(freq.iloc[0]) if len(freq) > 0 else 0,
    }


def all_categorical_stats(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Genera estadísticas para todas las columnas categóricas.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    results = []
    for col in cat_cols:
        stats = categorical_stats(df, col, top_n=top_n)
        if "error" not in stats:
            results.append(stats)
    return results


# ─────────────────────────────────────────────
# Estadísticas temporales
# ─────────────────────────────────────────────

def temporal_stats(df: pd.DataFrame, date_column: str) -> dict[str, Any]:
    """
    Calcula estadísticas de distribución temporal para una columna de fecha.
    """
    series = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if len(series) == 0:
        return {"error": f"La columna '{date_column}' no tiene fechas válidas."}

    return {
        "columna": date_column,
        "total_registros": len(series),
        "fecha_min": series.min().strftime("%Y-%m-%d"),
        "fecha_max": series.max().strftime("%Y-%m-%d"),
        "rango_dias": int((series.max() - series.min()).days),
        "por_año": series.dt.year.value_counts().sort_index().to_dict(),
        "por_mes": series.dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
        "por_dia_semana": series.dt.day_name().value_counts().to_dict(),
    }


def time_series_agg(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    freq: str = "ME",
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Agrega una variable numérica por período temporal.

    Parameters
    ----------
    freq     : 'D' (día), 'W' (semana), 'ME' (mes), 'QE' (trimestre), 'YE' (año).
    agg_func : 'sum', 'mean', 'count', 'median'.
    """
    df_temp = df[[date_column, value_column]].copy()
    df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors="coerce")
    df_temp = df_temp.dropna()
    df_temp = df_temp.set_index(date_column)
    df_temp = df_temp.sort_index()

    agg = df_temp[value_column].resample(freq).agg(agg_func)
    result = agg.reset_index()
    result.columns = ["fecha", value_column]
    return result


# ─────────────────────────────────────────────
# Correlaciones
# ─────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre columnas numéricas.

    Parameters
    ----------
    method : 'pearson', 'spearman' o 'kendall'.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) < 2:
        return pd.DataFrame()

    return df[num_cols].corr(method=method).round(4)


def top_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Devuelve las correlaciones más fuertes en formato de tabla plana.
    """
    corr = correlation_matrix(df, method=method)
    if corr.empty:
        return pd.DataFrame()

    # Convertir a formato largo y eliminar duplicados
    corr_long = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_long.columns = ["Variable A", "Variable B", "Correlación"]
    corr_long["Valor Absoluto"] = corr_long["Correlación"].abs()
    corr_long = corr_long.sort_values("Valor Absoluto", ascending=False).head(top_n)
    return corr_long.drop("Valor Absoluto", axis=1).reset_index(drop=True)


# ─────────────────────────────────────────────
# Resumen de columna (adaptativo)
# ─────────────────────────────────────────────

def column_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """
    Devuelve un resumen apropiado según el tipo de dato de la columna.
    """
    dtype = str(df[column].dtype)

    if pd.api.types.is_numeric_dtype(df[column]):
        return {"tipo": "numérico", "stats": numeric_stats(df, column)}

    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        return {"tipo": "fecha", "stats": temporal_stats(df, column)}

    else:
        return {"tipo": "categórico", "stats": categorical_stats(df, column)}


# ─────────────────────────────────────────────
# Resumen global del EDA
# ─────────────────────────────────────────────

def full_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Genera un resumen completo de EDA para todos los tipos de columnas.
    """
    return {
        "num_estadisticas": all_numeric_stats(df).to_dict() if not all_numeric_stats(df).empty else {},
        "cat_estadisticas": all_categorical_stats(df),
        "correlaciones": correlation_matrix(df).to_dict() if not correlation_matrix(df).empty else {},
        "top_correlaciones": top_correlations(df).to_dict(orient="records"),
    }
