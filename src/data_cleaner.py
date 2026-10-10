"""
data_cleaner.py
---------------
Módulo de limpieza y transformación de datos.

Proporciona operaciones individuales que pueden ejecutarse sobre una copia del DataFrame
sin modificar el original. Cada función devuelve el DataFrame transformado y un registro
de la operación realizada.

Operaciones disponibles:
    - remove_duplicates         : Elimina filas duplicadas.
    - handle_missing            : Gestiona valores faltantes.
    - convert_column_type       : Convierte tipo de dato de una columna.
    - convert_to_datetime       : Convierte columna a fecha.
    - strip_whitespace          : Elimina espacios innecesarios.
    - standardize_text          : Estandariza categorías de texto.
    - drop_columns              : Elimina columnas seleccionadas.
    - filter_rows               : Filtra filas por condición.
    - select_columns            : Conserva sólo las columnas indicadas.
    - handle_outliers           : Trata valores extremos.
    - rename_columns            : Renombra columnas.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import OUTLIER_IQR_FACTOR, OUTLIER_ZSCORE_THRESHOLD


# ─────────────────────────────────────────────
# Tipo para registros de historial
# ─────────────────────────────────────────────
OperationRecord = dict[str, Any]


# ─────────────────────────────────────────────
# 1. Duplicados
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Elimina filas duplicadas del DataFrame.

    Returns
    -------
    df_clean : DataFrame sin duplicados.
    record   : registro de la operación.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df_clean)
    removed = n_before - n_after

    record = {
        "operacion": "Eliminar duplicados",
        "filas_antes": n_before,
        "filas_despues": n_after,
        "afectados": removed,
        "parametros": {},
        "descripcion": f"Se eliminaron {removed} fila(s) duplicada(s).",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 2. Valores faltantes
# ─────────────────────────────────────────────

def handle_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = "eliminar_filas",
    fill_value: Any = None,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Gestiona valores faltantes en una columna específica.

    Parameters
    ----------
    column   : nombre de la columna a procesar.
    strategy : una de {'eliminar_filas', 'media', 'mediana', 'moda',
                        'valor_constante', 'ffill', 'bfill', 'conservar'}.
    fill_value : valor de relleno cuando strategy='valor_constante'.
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame.")

    df_clean = df.copy()

    # Validar estrategias numéricas ANTES del early-return, siempre
    if strategy in ("media", "mediana") and not pd.api.types.is_numeric_dtype(df_clean[column]):
        raise ValueError(
            f"La columna '{column}' no es numérica. "
            f"No se puede usar la estrategia '{strategy}'."
        )

    missing_before = int(df_clean[column].isna().sum())

    if missing_before == 0:
        record = {
            "operacion": "Gestionar faltantes",
            "columna": column,
            "afectados": 0,
            "parametros": {"estrategia": strategy},
            "descripcion": f"La columna '{column}' no tenía valores faltantes.",
        }
        return df_clean, record

    if strategy == "eliminar_filas":
        df_clean = df_clean.dropna(subset=[column]).reset_index(drop=True)
        desc = f"Se eliminaron {missing_before} fila(s) con valores faltantes en '{column}'."

    elif strategy == "media":
        if not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise ValueError(f"La columna '{column}' no es numérica. No se puede usar 'media'.")
        val = df_clean[column].mean()
        df_clean[column] = df_clean[column].fillna(val)
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con la media ({val:.4f})."

    elif strategy == "mediana":
        if not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise ValueError(f"La columna '{column}' no es numérica. No se puede usar 'mediana'.")
        val = df_clean[column].median()
        df_clean[column] = df_clean[column].fillna(val)
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con la mediana ({val:.4f})."

    elif strategy == "moda":
        mode_vals = df_clean[column].mode()
        if len(mode_vals) == 0:
            raise ValueError(f"No se pudo calcular la moda de '{column}'.")
        val = mode_vals.iloc[0]
        df_clean[column] = df_clean[column].fillna(val)
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con la moda ('{val}')."

    elif strategy == "valor_constante":
        if fill_value is None:
            raise ValueError("Se debe proporcionar 'fill_value' con strategy='valor_constante'.")
        df_clean[column] = df_clean[column].fillna(fill_value)
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con el valor '{fill_value}'."

    elif strategy == "ffill":
        df_clean[column] = df_clean[column].ffill()
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con el valor anterior (ffill)."

    elif strategy == "bfill":
        df_clean[column] = df_clean[column].bfill()
        desc = f"'{column}': {missing_before} faltante(s) reemplazados con el valor siguiente (bfill)."

    elif strategy == "conservar":
        desc = f"'{column}': Se conservaron {missing_before} faltante(s) sin modificación."

    else:
        raise ValueError(f"Estrategia desconocida: '{strategy}'.")

    record = {
        "operacion": "Gestionar faltantes",
        "columna": column,
        "afectados": missing_before,
        "parametros": {"estrategia": strategy, "valor": fill_value},
        "descripcion": desc,
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 3. Conversión de tipo de dato
# ─────────────────────────────────────────────

def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Convierte una columna al tipo de dato indicado.

    Parameters
    ----------
    target_type : 'int', 'float', 'str', 'bool', 'category'.
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe.")

    df_clean = df.copy()
    original_type = str(df_clean[column].dtype)
    errors_count = 0

    try:
        if target_type == "int":
            df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
            errors_count = int(df_clean[column].isna().sum() - df[column].isna().sum())
            df_clean[column] = df_clean[column].astype("Int64")  # nullable int

        elif target_type == "float":
            df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
            errors_count = int(df_clean[column].isna().sum() - df[column].isna().sum())

        elif target_type == "str":
            df_clean[column] = df_clean[column].astype(str)

        elif target_type == "bool":
            df_clean[column] = df_clean[column].astype(bool)

        elif target_type == "category":
            df_clean[column] = df_clean[column].astype("category")

        else:
            raise ValueError(f"Tipo '{target_type}' no soportado.")

    except Exception as exc:
        raise ValueError(
            f"No se pudo convertir '{column}' a '{target_type}': {exc}"
        ) from exc

    desc = (
        f"'{column}': {original_type} → {target_type}."
        + (f" {errors_count} valores no convertibles se marcaron como NaN." if errors_count > 0 else "")
    )
    record = {
        "operacion": "Convertir tipo de dato",
        "columna": column,
        "afectados": 1,
        "parametros": {"tipo_original": original_type, "tipo_nuevo": target_type},
        "descripcion": desc,
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 4. Conversión a fecha
# ─────────────────────────────────────────────

def convert_to_datetime(
    df: pd.DataFrame,
    column: str,
    date_format: Optional[str] = None,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Convierte una columna de texto a tipo fecha (datetime).
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe.")

    df_clean = df.copy()
    before_nulls = int(df_clean[column].isna().sum())

    try:
        if date_format:
            df_clean[column] = pd.to_datetime(
                df_clean[column], format=date_format, errors="coerce"
            )
        else:
            df_clean[column] = pd.to_datetime(
                df_clean[column], infer_datetime_format=True, errors="coerce"
            )
    except Exception as exc:
        raise ValueError(f"Error al convertir '{column}' a fecha: {exc}") from exc

    after_nulls = int(df_clean[column].isna().sum())
    failed = max(0, after_nulls - before_nulls)

    record = {
        "operacion": "Convertir a fecha",
        "columna": column,
        "afectados": int(len(df_clean) - failed),
        "parametros": {"formato": date_format or "automático"},
        "descripcion": (
            f"'{column}' convertida a datetime. "
            + (f"{failed} valores no convertibles." if failed > 0 else "Conversión completa.")
        ),
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 5. Eliminar espacios
# ─────────────────────────────────────────────

def strip_whitespace(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strip_column_names: bool = True,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Elimina espacios al inicio y final en columnas de texto y nombres de columnas.
    """
    df_clean = df.copy()
    affected_cells = 0

    if strip_column_names:
        df_clean.columns = [str(c).strip() for c in df_clean.columns]

    target_cols = columns if columns else df_clean.select_dtypes(include=["object"]).columns.tolist()

    for col in target_cols:
        if col in df_clean.columns and pd.api.types.is_object_dtype(df_clean[col]):
            before = df_clean[col].copy()
            df_clean[col] = df_clean[col].str.strip()
            affected_cells += int((df_clean[col] != before).sum())

    record = {
        "operacion": "Eliminar espacios",
        "columna": ", ".join(target_cols) if target_cols else "todas",
        "afectados": affected_cells,
        "parametros": {"columnas_nombre": strip_column_names},
        "descripcion": f"Se eliminaron espacios en {affected_cells} celda(s).",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 6. Estandarizar texto
# ─────────────────────────────────────────────

def standardize_text(
    df: pd.DataFrame,
    column: str,
    case: str = "lower",
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Estandariza el texto de una columna.

    Parameters
    ----------
    case : 'lower', 'upper', 'title'.
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe.")
    if not pd.api.types.is_object_dtype(df[column]):
        raise ValueError(f"La columna '{column}' no es de tipo texto.")

    df_clean = df.copy()
    if case == "lower":
        df_clean[column] = df_clean[column].str.lower()
    elif case == "upper":
        df_clean[column] = df_clean[column].str.upper()
    elif case == "title":
        df_clean[column] = df_clean[column].str.title()
    else:
        raise ValueError(f"Caso '{case}' no reconocido. Usa 'lower', 'upper' o 'title'.")

    record = {
        "operacion": "Estandarizar texto",
        "columna": column,
        "afectados": int(df_clean[column].notna().sum()),
        "parametros": {"caso": case},
        "descripcion": f"'{column}' convertida a '{case}'.",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 7. Eliminar columnas
# ─────────────────────────────────────────────

def drop_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, OperationRecord]:
    """Elimina las columnas indicadas."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas: {missing}")

    df_clean = df.drop(columns=columns)
    record = {
        "operacion": "Eliminar columnas",
        "columna": ", ".join(columns),
        "afectados": len(columns),
        "parametros": {"columnas": columns},
        "descripcion": f"Se eliminaron {len(columns)} columna(s): {', '.join(columns)}.",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 8. Filtrar filas
# ─────────────────────────────────────────────

def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Filtra filas según una condición simple.

    Parameters
    ----------
    operator : '==', '!=', '>', '>=', '<', '<=', 'contiene', 'no_contiene'.
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe.")

    n_before = len(df)
    df_clean = df.copy()

    try:
        if operator == "==":
            mask = df_clean[column] == value
        elif operator == "!=":
            mask = df_clean[column] != value
        elif operator == ">":
            mask = df_clean[column] > value
        elif operator == ">=":
            mask = df_clean[column] >= value
        elif operator == "<":
            mask = df_clean[column] < value
        elif operator == "<=":
            mask = df_clean[column] <= value
        elif operator == "contiene":
            mask = df_clean[column].astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "no_contiene":
            mask = ~df_clean[column].astype(str).str.contains(str(value), case=False, na=False)
        else:
            raise ValueError(f"Operador '{operator}' no reconocido.")

        df_clean = df_clean[mask].reset_index(drop=True)
    except Exception as exc:
        raise ValueError(f"Error al aplicar el filtro: {exc}") from exc

    removed = n_before - len(df_clean)
    record = {
        "operacion": "Filtrar filas",
        "columna": column,
        "afectados": removed,
        "parametros": {"operador": operator, "valor": value},
        "descripcion": (
            f"Filtro '{column} {operator} {value}': se conservaron "
            f"{len(df_clean)} fila(s), se eliminaron {removed}."
        ),
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 9. Seleccionar columnas
# ─────────────────────────────────────────────

def select_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, OperationRecord]:
    """Conserva únicamente las columnas indicadas."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas: {missing}")

    df_clean = df[columns].copy()
    dropped = df.shape[1] - len(columns)
    record = {
        "operacion": "Seleccionar columnas",
        "columna": "múltiples",
        "afectados": dropped,
        "parametros": {"columnas": columns},
        "descripcion": f"Se conservaron {len(columns)} columna(s). Se descartaron {dropped}.",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 10. Tratamiento de valores extremos
# ─────────────────────────────────────────────

def handle_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    action: str = "eliminar",
    cap_lower: Optional[float] = None,
    cap_upper: Optional[float] = None,
) -> tuple[pd.DataFrame, OperationRecord]:
    """
    Trata valores extremos en una columna numérica.

    Parameters
    ----------
    method : 'iqr' o 'zscore'.
    action : 'eliminar' o 'capturar' (winsorize).
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La columna '{column}' no es numérica.")

    df_clean = df.copy()
    series = df_clean[column].dropna()
    n_before = len(df_clean)

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = cap_lower if cap_lower is not None else q1 - OUTLIER_IQR_FACTOR * iqr
        upper = cap_upper if cap_upper is not None else q3 + OUTLIER_IQR_FACTOR * iqr
    elif method == "zscore":
        from scipy import stats as scipy_stats
        z = np.abs(scipy_stats.zscore(series))
        lower = float(series[z <= OUTLIER_ZSCORE_THRESHOLD].min())
        upper = float(series[z <= OUTLIER_ZSCORE_THRESHOLD].max())
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa 'iqr' o 'zscore'.")

    outlier_mask = (df_clean[column] < lower) | (df_clean[column] > upper)
    n_outliers = int(outlier_mask.sum())

    if action == "eliminar":
        df_clean = df_clean[~outlier_mask].reset_index(drop=True)
        desc = f"Se eliminaron {n_outliers} fila(s) con valores extremos en '{column}'."
    elif action == "capturar":
        df_clean[column] = df_clean[column].clip(lower=lower, upper=upper)
        desc = f"Se capturaron {n_outliers} valor(es) extremo(s) en '{column}' al rango [{lower:.4f}, {upper:.4f}]."
    else:
        raise ValueError(f"Acción '{action}' no reconocida. Usa 'eliminar' o 'capturar'.")

    record = {
        "operacion": "Tratar valores extremos",
        "columna": column,
        "afectados": n_outliers,
        "parametros": {
            "método": method, "acción": action,
            "límite_inferior": lower, "límite_superior": upper,
        },
        "descripcion": desc,
    }
    return df_clean, record


# ─────────────────────────────────────────────
# 11. Renombrar columnas
# ─────────────────────────────────────────────

def rename_columns(
    df: pd.DataFrame,
    rename_map: dict[str, str],
) -> tuple[pd.DataFrame, OperationRecord]:
    """Renombra columnas según el mapa proporcionado."""
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas para renombrar: {missing}")

    df_clean = df.rename(columns=rename_map)
    record = {
        "operacion": "Renombrar columnas",
        "columna": ", ".join(rename_map.keys()),
        "afectados": len(rename_map),
        "parametros": {"mapa": rename_map},
        "descripcion": f"Se renombraron {len(rename_map)} columna(s).",
    }
    return df_clean, record


# ─────────────────────────────────────────────
# Gestión del historial de operaciones
# ─────────────────────────────────────────────

class CleaningHistory:
    """
    Mantiene un historial de operaciones de limpieza y permite deshacer la última.
    """

    def __init__(self, original_df: pd.DataFrame):
        self._original = original_df.copy()
        self._snapshots: list[pd.DataFrame] = [original_df.copy()]
        self._records: list[OperationRecord] = []

    def apply(
        self,
        func,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, OperationRecord]:
        """
        Aplica una función de limpieza sobre el estado actual y guarda la operación.
        """
        current = self._snapshots[-1]
        new_df, record = func(current, *args, **kwargs)
        self._snapshots.append(new_df)
        self._records.append(record)
        return new_df, record

    def undo(self) -> pd.DataFrame:
        """Revierte la última operación."""
        if len(self._snapshots) <= 1:
            raise ValueError("No hay operaciones que deshacer.")
        self._snapshots.pop()
        self._records.pop()
        return self._snapshots[-1].copy()

    def reset(self) -> pd.DataFrame:
        """Vuelve al DataFrame original."""
        self._snapshots = [self._original.copy()]
        self._records = []
        return self._original.copy()

    @property
    def current(self) -> pd.DataFrame:
        return self._snapshots[-1].copy()

    @property
    def history(self) -> list[OperationRecord]:
        return list(self._records)

    @property
    def can_undo(self) -> bool:
        return len(self._snapshots) > 1
