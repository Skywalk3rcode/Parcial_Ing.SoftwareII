"""
data_loader.py
--------------
Módulo responsable de leer archivos CSV y Excel con validación robusta.

Funciones principales:
    - load_csv: lee un CSV con separador y codificación configurable.
    - load_excel: lee una hoja específica de un archivo Excel.
    - validate_dataframe: comprueba que el DataFrame sea utilizable.
    - get_file_info: devuelve metadatos del archivo cargado.
"""

import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Agrega el directorio raíz al path para permitir importaciones relativas
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    MAX_FILE_SIZE_MB,
    MAX_ROWS_WARN,
    CSV_ENCODINGS,
    PREVIEW_ROWS,
)


# ─────────────────────────────────────────────
# Carga de CSV
# ─────────────────────────────────────────────

def load_csv(
    file_obj,
    separator: str = ",",
    encoding: str = "utf-8",
    decimal: str = ".",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Lee un archivo CSV y devuelve un DataFrame junto con advertencias.

    Parameters
    ----------
    file_obj : archivo cargado (UploadedFile de Streamlit o ruta de archivo).
    separator : separador de columnas.
    encoding  : codificación del archivo.
    decimal   : carácter decimal.

    Returns
    -------
    df       : DataFrame resultante.
    warnings : lista de mensajes de advertencia (puede estar vacía).
    """
    warnings: list[str] = []

    # Leer contenido en bytes para reutilizarlo en caso de error
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        with open(file_obj, "rb") as f:
            raw = f.read()

    # Verificar tamaño
    size_mb = len(raw) / (1024 ** 2)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB "
            f"({size_mb:.1f} MB). Por favor, usa un archivo más pequeño."
        )

    # Intentar decodificar con la codificación solicitada o autodetectar
    encodings_to_try = [encoding] + [e for e in CSV_ENCODINGS if e != encoding]
    df = None
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=separator,
                encoding=enc,
                decimal=decimal,
                on_bad_lines="warn",
                low_memory=False,
            )
            if enc != encoding:
                warnings.append(
                    f"El archivo no pudo leerse con '{encoding}'. "
                    f"Se usó '{enc}' automáticamente."
                )
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise ValueError(f"Error inesperado al leer el archivo: {exc}") from exc

    if df is None:
        raise ValueError(
            f"No fue posible leer el archivo con ninguna codificación conocida. "
            f"Último error: {last_error}"
        )

    # Validar contenido
    validation_warnings = validate_dataframe(df)
    warnings.extend(validation_warnings)

    # Advertencia de tamaño
    if len(df) > MAX_ROWS_WARN:
        warnings.append(
            f"El archivo contiene {len(df):,} filas. "
            f"El procesamiento puede ser lento."
        )

    return df, warnings


# ─────────────────────────────────────────────
# Carga de Excel
# ─────────────────────────────────────────────

def load_excel(
    file_obj,
    sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Lee un archivo Excel y devuelve el DataFrame, advertencias y lista de hojas.

    Parameters
    ----------
    file_obj   : archivo cargado (UploadedFile o ruta).
    sheet_name : nombre de la hoja; si es None se usa la primera.

    Returns
    -------
    df         : DataFrame resultante.
    warnings   : lista de advertencias.
    sheet_list : lista de nombres de hojas disponibles.
    """
    warnings: list[str] = []

    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        with open(file_obj, "rb") as f:
            raw = f.read()

    size_mb = len(raw) / (1024 ** 2)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB ({size_mb:.1f} MB)."
        )

    try:
        excel_file = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el archivo Excel: {exc}") from exc

    sheet_list: list[str] = excel_file.sheet_names

    if not sheet_list:
        raise ValueError("El archivo Excel no contiene hojas válidas.")

    target_sheet = sheet_name if sheet_name in sheet_list else sheet_list[0]
    if sheet_name and sheet_name not in sheet_list:
        warnings.append(
            f"La hoja '{sheet_name}' no existe. Se cargó '{target_sheet}'."
        )

    try:
        df = pd.read_excel(
            io.BytesIO(raw),
            sheet_name=target_sheet,
            engine="openpyxl",
        )
    except Exception as exc:
        raise ValueError(
            f"Error al leer la hoja '{target_sheet}': {exc}"
        ) from exc

    validation_warnings = validate_dataframe(df)
    warnings.extend(validation_warnings)

    if len(df) > MAX_ROWS_WARN:
        warnings.append(
            f"El archivo contiene {len(df):,} filas. El procesamiento puede ser lento."
        )

    return df, warnings, sheet_list


# ─────────────────────────────────────────────
# Validación
# ─────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Comprueba que el DataFrame tenga contenido utilizable.

    Returns
    -------
    warnings : lista de advertencias sobre problemas detectados.

    Raises
    ------
    ValueError : cuando el DataFrame no es utilizable.
    """
    warnings: list[str] = []

    if df is None:
        raise ValueError("No se pudo cargar el archivo.")

    if df.empty:
        raise ValueError("El archivo está vacío: no contiene filas de datos.")

    if df.shape[1] == 0:
        raise ValueError("El archivo no contiene columnas válidas.")

    if df.shape[1] == 1:
        warnings.append(
            "El archivo sólo tiene una columna. "
            "Verifica que el separador sea el correcto."
        )

    # Detectar si todas las columnas son 'Unnamed'
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if len(unnamed) == df.shape[1]:
        warnings.append(
            "Todas las columnas tienen nombres genéricos. "
            "Es posible que la primera fila no sea el encabezado."
        )

    # Detectar columnas completamente vacías
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(
            f"{len(empty_cols)} columna(s) están completamente vacías: "
            f"{', '.join(str(c) for c in empty_cols[:5])}."
        )

    return warnings


# ─────────────────────────────────────────────
# Metadatos del archivo
# ─────────────────────────────────────────────

def get_file_info(file_obj, df: pd.DataFrame) -> dict:
    """
    Construye un diccionario con metadatos del archivo y del DataFrame.

    Parameters
    ----------
    file_obj : archivo original cargado.
    df       : DataFrame ya leído.

    Returns
    -------
    info : dict con nombre, tamaño, filas, columnas, etc.
    """
    try:
        file_name = getattr(file_obj, "name", str(file_obj))
        file_size = getattr(file_obj, "size", 0) or 0
        size_mb = file_size / (1024 ** 2)
    except Exception:
        file_name = "desconocido"
        size_mb = 0.0

    # Clasificar columnas por tipo
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    missing_total = int(df.isna().sum().sum())
    missing_pct = round(missing_total / max(df.size, 1) * 100, 2)

    return {
        "nombre_archivo": file_name,
        "tamaño_mb": round(size_mb, 3),
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_numericas": len(numeric_cols),
        "columnas_categoricas": len(cat_cols),
        "columnas_fecha": len(date_cols),
        "valores_faltantes": missing_total,
        "porcentaje_faltantes": missing_pct,
        "nombres_columnas": df.columns.tolist(),
        "tipos_columnas": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "columnas_numericas_lista": numeric_cols,
        "columnas_categoricas_lista": cat_cols,
        "columnas_fecha_lista": date_cols,
    }
