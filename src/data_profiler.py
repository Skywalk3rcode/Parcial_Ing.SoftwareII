"""
data_profiler.py
----------------
Módulo de perfilado y evaluación de calidad de datos.

Genera un informe detallado con:
    - Valores faltantes por columna.
    - Filas duplicadas.
    - Valores únicos.
    - Posibles columnas identificadoras.
    - Valores numéricos almacenados como texto.
    - Fechas como cadenas de texto.
    - Valores extremos (IQR y Z-score).
    - Inconsistencias de formato.
    - Tabla de calidad resumida.
"""

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    MISSING_WARN_THRESHOLD,
    MISSING_HIGH_THRESHOLD,
    OUTLIER_IQR_FACTOR,
    OUTLIER_ZSCORE_THRESHOLD,
)


# ─────────────────────────────────────────────
# Funciones de utilidad interna
# ─────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{2}/\d{2}/\d{4}",
    r"\d{2}-\d{2}-\d{4}",
    r"\d{1,2}/\d{1,2}/\d{2,4}",
]


def _looks_like_date(series: pd.Series, sample: int = 200) -> bool:
    """Comprueba si una columna de texto parece contener fechas."""
    sample_vals = series.dropna().astype(str).head(sample)
    if len(sample_vals) == 0:
        return False
    pattern = re.compile("|".join(_DATE_PATTERNS))
    matches = sample_vals.apply(lambda x: bool(pattern.search(x)))
    return matches.mean() > 0.7


def _looks_like_numeric(series: pd.Series, sample: int = 200) -> bool:
    """Comprueba si una columna de texto parece contener números."""
    sample_vals = series.dropna().astype(str).head(sample)
    if len(sample_vals) == 0:
        return False

    def is_num(v: str) -> bool:
        try:
            float(v.replace(",", ".").replace(" ", ""))
            return True
        except ValueError:
            return False

    return sample_vals.apply(is_num).mean() > 0.8


def _detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Detecta valores extremos mediante el método IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - OUTLIER_IQR_FACTOR * iqr
    upper = q3 + OUTLIER_IQR_FACTOR * iqr
    return series[(series < lower) | (series > upper)]


def _detect_outliers_zscore(series: pd.Series) -> pd.Series:
    """Detecta valores extremos mediante Z-score."""
    z = np.abs(stats.zscore(series.dropna()))
    return series.dropna()[z > OUTLIER_ZSCORE_THRESHOLD]


# ─────────────────────────────────────────────
# Función principal de perfilado
# ─────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Genera un perfil completo de calidad del DataFrame.

    Parameters
    ----------
    df : DataFrame a analizar.

    Returns
    -------
    profile : dict con todas las métricas de calidad.
    """
    profile: dict[str, Any] = {}

    # 1. Valores faltantes
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    profile["missing"] = pd.DataFrame({
        "Columna": missing_count.index,
        "Faltantes": missing_count.values,
        "Porcentaje (%)": missing_pct.values,
        "Nivel": missing_pct.apply(_classify_missing),
    }).reset_index(drop=True)

    # 2. Duplicados
    dup_count = int(df.duplicated().sum())
    profile["duplicates"] = {
        "cantidad": dup_count,
        "porcentaje": round(dup_count / max(len(df), 1) * 100, 2),
    }

    # 3. Valores únicos
    unique_counts = df.nunique()
    profile["unique_values"] = pd.DataFrame({
        "Columna": unique_counts.index,
        "Únicos": unique_counts.values,
        "Porcentaje (%)": (unique_counts / len(df) * 100).round(2).values,
    }).reset_index(drop=True)

    # 4. Posibles columnas identificadoras (cardinalidad alta)
    id_cols = [
        col for col in df.columns
        if df[col].nunique() == len(df) or (
            df[col].nunique() / len(df) > 0.95 and len(df) > 50
        )
    ]
    profile["possible_id_columns"] = id_cols

    # 5. Tipos de dato y posibles inconsistencias
    profile["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
    profile["dtype_suggestions"] = _suggest_dtype_changes(df)

    # 6. Valores numéricos almacenados como texto
    text_cols = df.select_dtypes(include=["object"]).columns
    profile["numeric_as_text"] = [
        col for col in text_cols if _looks_like_numeric(df[col])
    ]

    # 7. Fechas almacenadas como cadenas
    profile["dates_as_text"] = [
        col for col in text_cols
        if col not in profile["numeric_as_text"] and _looks_like_date(df[col])
    ]

    # 8. Valores extremos por columna numérica
    num_cols = df.select_dtypes(include=["number"]).columns
    outlier_info = {}
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        iqr_outliers = _detect_outliers_iqr(series)
        z_outliers = _detect_outliers_zscore(series)
        if len(iqr_outliers) > 0:
            outlier_info[col] = {
                "iqr_count": len(iqr_outliers),
                "iqr_min": float(iqr_outliers.min()),
                "iqr_max": float(iqr_outliers.max()),
                "z_count": len(z_outliers),
            }
    profile["outliers"] = outlier_info

    # 9. Inconsistencias de formato en columnas de texto
    profile["format_issues"] = _detect_format_issues(df)

    # 10. Tabla de calidad resumida
    profile["quality_table"] = _build_quality_table(df, profile)

    # 11. Puntuación de calidad global (0–100)
    profile["quality_score"] = _compute_quality_score(df, profile)

    return profile


# ─────────────────────────────────────────────
# Funciones auxiliares de perfilado
# ─────────────────────────────────────────────

def _classify_missing(pct: float) -> str:
    if pct == 0:
        return "✅ Sin faltantes"
    elif pct < MISSING_WARN_THRESHOLD * 100:
        return "🟡 Bajo"
    elif pct < MISSING_HIGH_THRESHOLD * 100:
        return "🟠 Moderado"
    else:
        return "🔴 Alto"


def _suggest_dtype_changes(df: pd.DataFrame) -> list[dict]:
    """Sugiere conversiones de tipo de dato para columnas con tipos inapropiados."""
    suggestions = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype == "object":
            if _looks_like_numeric(df[col]):
                suggestions.append({
                    "columna": col,
                    "tipo_actual": dtype,
                    "tipo_sugerido": "numérico (float)",
                    "motivo": "La mayoría de valores son números"
                })
            elif _looks_like_date(df[col]):
                suggestions.append({
                    "columna": col,
                    "tipo_actual": dtype,
                    "tipo_sugerido": "fecha (datetime)",
                    "motivo": "La mayoría de valores parecen fechas"
                })
        elif dtype in ("float64", "float32"):
            # Posible conversión a entero si no hay decimales reales
            series = df[col].dropna()
            if len(series) > 0 and (series == series.astype(int).astype(float)).all():
                suggestions.append({
                    "columna": col,
                    "tipo_actual": dtype,
                    "tipo_sugerido": "entero (int)",
                    "motivo": "Los valores no tienen decimales reales"
                })
    return suggestions


def _detect_format_issues(df: pd.DataFrame) -> list[dict]:
    """Detecta inconsistencias de formato en columnas de texto."""
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        # Espacios al inicio o al final
        with_spaces = series.apply(lambda x: x != x.strip()).sum()
        if with_spaces > 0:
            issues.append({
                "columna": col,
                "problema": "Espacios al inicio/final",
                "afectados": int(with_spaces),
            })

        # Mezcla de mayúsculas y minúsculas
        lower_count = series.apply(lambda x: x == x.lower()).sum()
        upper_count = series.apply(lambda x: x == x.upper()).sum()
        mixed = len(series) - lower_count - upper_count
        if mixed > len(series) * 0.1 and lower_count > 0 and upper_count > 0:
            issues.append({
                "columna": col,
                "problema": "Mezcla inconsistente de mayúsculas/minúsculas",
                "afectados": int(mixed),
            })

    return issues


def _build_quality_table(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Construye una tabla de calidad por columna."""
    rows = []
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df) * 100)
        has_outliers = col in profile.get("outliers", {})
        is_numeric_text = col in profile.get("numeric_as_text", [])
        is_date_text = col in profile.get("dates_as_text", [])
        is_id = col in profile.get("possible_id_columns", [])
        fmt_issues = [i for i in profile.get("format_issues", []) if i["columna"] == col]

        problems = []
        if missing_pct > 0:
            problems.append(f"Faltantes: {missing_pct:.1f}%")
        if has_outliers:
            n = profile["outliers"][col]["iqr_count"]
            problems.append(f"Extremos: {n}")
        if is_numeric_text:
            problems.append("Números como texto")
        if is_date_text:
            problems.append("Fechas como texto")
        if is_id:
            problems.append("Posible ID")
        for fi in fmt_issues:
            problems.append(fi["problema"])

        rows.append({
            "Columna": col,
            "Tipo": str(df[col].dtype),
            "Faltantes (%)": round(missing_pct, 1),
            "Únicos": int(df[col].nunique()),
            "Problemas detectados": " | ".join(problems) if problems else "✅ Ninguno",
        })

    return pd.DataFrame(rows)


def _compute_quality_score(df: pd.DataFrame, profile: dict) -> float:
    """Calcula un índice de calidad del 0 al 100."""
    score = 100.0

    # Penalizar por faltantes
    missing_pct = df.isna().sum().sum() / max(df.size, 1) * 100
    score -= min(missing_pct * 1.5, 30)

    # Penalizar por duplicados
    dup_pct = profile["duplicates"]["porcentaje"]
    score -= min(dup_pct * 0.5, 15)

    # Penalizar por columnas con tipo incorrecto
    type_issues = len(profile["numeric_as_text"]) + len(profile["dates_as_text"])
    score -= min(type_issues * 3, 20)

    # Penalizar por inconsistencias de formato
    fmt_issues = len(profile["format_issues"])
    score -= min(fmt_issues * 2, 15)

    return round(max(score, 0), 1)


# ─────────────────────────────────────────────
# Resumen rápido de calidad
# ─────────────────────────────────────────────

def quick_quality_summary(df: pd.DataFrame) -> dict:
    """
    Devuelve un resumen rápido de calidad sin el análisis completo.
    Útil para mostrar indicadores en el dashboard.
    """
    missing_total = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())
    missing_cols = int((df.isna().sum() > 0).sum())

    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "valores_faltantes": missing_total,
        "porcentaje_faltantes": round(missing_total / max(df.size, 1) * 100, 2),
        "columnas_con_faltantes": missing_cols,
        "filas_duplicadas": dup_count,
        "columnas_numericas": len(df.select_dtypes(include=["number"]).columns),
        "columnas_categoricas": len(df.select_dtypes(include=["object", "category"]).columns),
        "columnas_fecha": len(df.select_dtypes(include=["datetime"]).columns),
    }
