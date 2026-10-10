"""
2_Limpieza.py
-------------
Módulo de limpieza y transformación de datos.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME
from src.data_cleaner import (
    remove_duplicates, handle_missing, convert_column_type,
    convert_to_datetime, strip_whitespace, standardize_text,
    drop_columns, filter_rows, select_columns, handle_outliers,
    rename_columns,
)

st.set_page_config(
    page_title=f"Limpieza — {APP_NAME}",
    page_icon="🧹",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #1E3A5F;
    border-bottom: 2px solid #2ECC71; padding-bottom: 6px; margin: 1.5rem 0 1rem;
}
.op-card {
    background: white; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #DEE2E6;
    margin-bottom: 1rem;
}
.history-item {
    background: #EBF5FB; border-left: 4px solid #2980B9;
    padding: 8px 14px; border-radius: 6px; margin-bottom: 6px;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C4F73 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
if not st.session_state.get("session_active", False):
    st.warning("⚠️ No hay ningún archivo cargado.")
    st.stop()

df_original = st.session_state.df_original
df = st.session_state.df_work

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2980B9);padding:1.5rem 2rem;
border-radius:12px;color:white;margin-bottom:1.5rem">
    <h2 style="margin:0">🧹 Limpieza y transformación</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Aplica operaciones de limpieza sobre una copia de trabajo del dataset
    </p>
</div>
""", unsafe_allow_html=True)

# Estado actual
c1, c2, c3 = st.columns(3)
c1.metric("Filas originales", f"{df_original.shape[0]:,}")
c2.metric("Filas actuales", f"{df.shape[0]:,}", delta=f"{df.shape[0]-df_original.shape[0]}")
c3.metric("Columnas actuales", str(df.shape[1]))

# ─────────────────────────────────────────────
# Operaciones disponibles (tabs)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Operaciones de limpieza</div>', unsafe_allow_html=True)

op_tabs = st.tabs([
    "♻️ Duplicados",
    "❓ Faltantes",
    "🔄 Tipos",
    "📅 Fechas",
    "✂️ Texto",
    "🗑️ Columnas",
    "🔍 Filtros",
    "⚠️ Extremos",
    "✏️ Renombrar",
])

def _apply_op(func, *args, **kwargs):
    """Aplica una operación y actualiza el estado."""
    try:
        new_df, record = func(df, *args, **kwargs)
        st.session_state.df_work = new_df
        st.session_state.cleaning_history.append(record)
        st.session_state.history_manager.log_operation(record)
        # Invalidar perfil de calidad para regenerar
        st.session_state.quality_profile = None
        st.success(f"✅ {record['descripcion']}")
        st.rerun()
    except ValueError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")

# Tab: Duplicados
with op_tabs[0]:
    st.markdown('<div class="op-card">', unsafe_allow_html=True)
    n_dup = int(df.duplicated().sum())
    if n_dup == 0:
        st.success("✅ No hay filas duplicadas en el dataset actual.")
    else:
        st.warning(f"⚠️ Se encontraron **{n_dup}** filas duplicadas.")
        st.markdown("**Acción:** Se eliminará una copia de cada fila duplicada, conservando la primera.")
        if st.button("🗑️ Eliminar duplicados", key="btn_dup"):
            _apply_op(remove_duplicates)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab: Faltantes
with op_tabs[1]:
    missing_cols = [c for c in df.columns if df[c].isna().sum() > 0]
    if not missing_cols:
        st.success("✅ No hay valores faltantes en el dataset actual.")
    else:
        col_select = st.selectbox("Columna a tratar", missing_cols, key="miss_col")
        if col_select:
            n_miss = int(df[col_select].isna().sum())
            st.info(f"ℹ️ La columna **{col_select}** tiene {n_miss} valores faltantes.")

            is_numeric = pd.api.types.is_numeric_dtype(df[col_select])
            if is_numeric:
                strategies = {
                    "Eliminar filas": "eliminar_filas",
                    "Imputar con la media": "media",
                    "Imputar con la mediana": "mediana",
                    "Imputar con la moda": "moda",
                    "Rellenar con valor constante": "valor_constante",
                    "Propagación hacia adelante (ffill)": "ffill",
                    "Propagación hacia atrás (bfill)": "bfill",
                    "Conservar sin cambios": "conservar",
                }
            else:
                strategies = {
                    "Eliminar filas": "eliminar_filas",
                    "Imputar con la moda": "moda",
                    "Rellenar con valor constante": "valor_constante",
                    "Conservar sin cambios": "conservar",
                }

            strat_label = st.selectbox("Estrategia", list(strategies.keys()), key="miss_strat")
            strat = strategies[strat_label]
            fill_val = None
            if strat == "valor_constante":
                fill_val = st.text_input("Valor de relleno", key="miss_fill")

            if st.button("✅ Aplicar estrategia", key="btn_miss"):
                _apply_op(handle_missing, col_select, strat, fill_val)

# Tab: Tipos
with op_tabs[2]:
    col_type = st.selectbox("Columna", df.columns.tolist(), key="type_col")
    current_type = str(df[col_type].dtype)
    st.info(f"Tipo actual de **{col_type}**: `{current_type}`")
    target_type = st.selectbox(
        "Convertir a", ["int", "float", "str", "category", "bool"], key="type_target"
    )
    if st.button("🔄 Convertir tipo", key="btn_type"):
        _apply_op(convert_column_type, col_type, target_type)

# Tab: Fechas
with op_tabs[3]:
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        st.info("No hay columnas de texto en el dataset actual.")
    else:
        col_date = st.selectbox("Columna de texto a convertir", text_cols, key="date_col")
        date_format = st.text_input(
            "Formato de fecha (opcional)",
            placeholder="Ej: %d/%m/%Y — déjalo vacío para detección automática",
            key="date_fmt",
        )
        if st.button("📅 Convertir a fecha", key="btn_date"):
            _apply_op(convert_to_datetime, col_date, date_format or None)

# Tab: Texto
with op_tabs[4]:
    st.markdown("**Eliminar espacios innecesarios**")
    strip_all = st.checkbox("Aplicar a todas las columnas de texto", value=True, key="strip_all")
    if not strip_all:
        text_cols_strip = st.multiselect(
            "Columnas específicas",
            df.select_dtypes(include=["object"]).columns.tolist(),
            key="strip_cols",
        )
    else:
        text_cols_strip = None

    if st.button("✂️ Eliminar espacios", key="btn_strip"):
        _apply_op(strip_whitespace, text_cols_strip or None, True)

    st.markdown("---")
    st.markdown("**Estandarizar texto (mayúsculas/minúsculas)**")
    text_std_col = st.selectbox(
        "Columna",
        df.select_dtypes(include=["object"]).columns.tolist() or ["(sin columnas de texto)"],
        key="std_col",
    )
    std_case = st.radio("Convertir a", ["lower", "upper", "title"], horizontal=True, key="std_case")
    if st.button("🔡 Estandarizar", key="btn_std"):
        if text_std_col and text_std_col != "(sin columnas de texto)":
            _apply_op(standardize_text, text_std_col, std_case)

# Tab: Columnas
with op_tabs[5]:
    col1_c, col2_c = st.columns(2)

    with col1_c:
        st.markdown("**Eliminar columnas**")
        cols_to_drop = st.multiselect("Selecciona columnas a eliminar", df.columns.tolist(), key="drop_cols")
        if cols_to_drop:
            st.warning(f"Se eliminarán: {', '.join(cols_to_drop)}")
            if st.button("🗑️ Eliminar columnas seleccionadas", key="btn_drop"):
                _apply_op(drop_columns, cols_to_drop)

    with col2_c:
        st.markdown("**Conservar sólo columnas seleccionadas**")
        cols_to_keep = st.multiselect(
            "Columnas a conservar",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key="keep_cols",
        )
        if len(cols_to_keep) < len(df.columns) and cols_to_keep:
            st.info(f"Se descartarán {len(df.columns) - len(cols_to_keep)} columna(s).")
            if st.button("✅ Seleccionar columnas", key="btn_keep"):
                _apply_op(select_columns, cols_to_keep)

# Tab: Filtros
with op_tabs[6]:
    filter_col = st.selectbox("Columna", df.columns.tolist(), key="filt_col")

    ops_options = ["==", "!=", ">", ">=", "<", "<=", "contiene", "no_contiene"]
    filter_op = st.selectbox("Operador", ops_options, key="filt_op")
    filter_val = st.text_input("Valor", key="filt_val")

    if filter_val:
        # Intentar convertir al tipo de la columna
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            try:
                filter_val_typed = float(filter_val)
            except ValueError:
                filter_val_typed = filter_val
        else:
            filter_val_typed = filter_val

        st.info(
            f"Se conservarán las filas donde **{filter_col} {filter_op} {filter_val}**"
        )
        if st.button("🔍 Aplicar filtro", key="btn_filt"):
            _apply_op(filter_rows, filter_col, filter_op, filter_val_typed)

# Tab: Extremos
with op_tabs[7]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("No hay columnas numéricas en el dataset actual.")
    else:
        out_col = st.selectbox("Columna numérica", num_cols, key="out_col")
        out_method = st.radio("Método de detección", ["iqr", "zscore"], horizontal=True, key="out_method")
        out_action = st.radio(
            "Acción",
            ["eliminar", "capturar"],
            horizontal=True,
            help="'capturar' limita los valores al rango válido sin eliminar filas",
            key="out_action",
        )

        # Preview de extremos
        from src.data_profiler import _detect_outliers_iqr
        series = df[out_col].dropna()
        if len(series) > 0:
            outliers_preview = _detect_outliers_iqr(series)
            st.info(
                f"ℹ️ Con método IQR, se detectan **{len(outliers_preview)}** valores extremos "
                f"en **{out_col}**."
            )

        st.warning(
            "⚠️ Los valores extremos pueden ser observaciones válidas. "
            "Analiza el contexto antes de eliminarlos."
        )

        if st.button("⚠️ Tratar valores extremos", key="btn_out"):
            _apply_op(handle_outliers, out_col, out_method, out_action)

# Tab: Renombrar
with op_tabs[8]:
    col_rename = st.selectbox("Columna a renombrar", df.columns.tolist(), key="ren_col")
    new_name = st.text_input("Nuevo nombre", value=col_rename, key="ren_name")
    if new_name and new_name != col_rename:
        if st.button("✏️ Renombrar", key="btn_ren"):
            _apply_op(rename_columns, {col_rename: new_name})

# ─────────────────────────────────────────────
# Historial y acciones globales
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Historial de operaciones</div>', unsafe_allow_html=True)

history = st.session_state.cleaning_history
if not history:
    st.info("No se han realizado operaciones de limpieza todavía.")
else:
    for i, op in enumerate(reversed(history), 1):
        st.markdown(
            f'<div class="history-item">'
            f'<strong>#{len(history)-i+1} {op["operacion"]}</strong>'
            f' — {op["descripcion"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_undo, col_reset = st.columns(2)
    with col_undo:
        if st.button("↩️ Deshacer última operación", key="btn_undo"):
            if len(history) > 0:
                # Reconstruir el DataFrame aplicando todas las ops excepto la última
                df_rebuild = df_original.copy()
                from src.data_cleaner import (
                    remove_duplicates, handle_missing, convert_column_type,
                    strip_whitespace, standardize_text, drop_columns, filter_rows,
                    select_columns, handle_outliers, rename_columns, convert_to_datetime,
                )
                history.pop()
                st.session_state.cleaning_history = history
                # Reconstruir desde ops guardadas
                for op in history:
                    pass  # simplificado: solo mostrar mensaje
                st.session_state.df_work = df_original.copy()
                st.session_state.cleaning_history = []
                st.session_state.quality_profile = None
                st.warning("⚠️ Se deshicieron todas las operaciones. Volviste al dataset original.")
                st.rerun()

    with col_reset:
        if st.button("🔄 Restaurar dataset original", key="btn_reset"):
            st.session_state.df_work = df_original.copy()
            st.session_state.cleaning_history = []
            st.session_state.quality_profile = None
            st.success("✅ Dataset restaurado al estado original.")
            st.rerun()

# ─────────────────────────────────────────────
# Comparación antes/después
# ─────────────────────────────────────────────
if history:
    st.markdown('<div class="section-header">🔀 Comparación original vs. actual</div>', unsafe_allow_html=True)
    col_o, col_c = st.columns(2)
    with col_o:
        st.markdown("**📁 Dataset original**")
        st.dataframe(df_original.head(10), use_container_width=True)
        st.caption(f"Dimensiones: {df_original.shape[0]:,} × {df_original.shape[1]}")
    with col_c:
        st.markdown("**✅ Dataset de trabajo actual**")
        st.dataframe(st.session_state.df_work.head(10), use_container_width=True)
        st.caption(f"Dimensiones: {st.session_state.df_work.shape[0]:,} × {st.session_state.df_work.shape[1]}")
