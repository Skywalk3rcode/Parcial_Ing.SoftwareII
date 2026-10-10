"""
app.py
------
Punto de entrada principal de DataInsight AI.

Muestra la página de inicio, el formulario de carga de archivos y
el dashboard de resumen general. El estado del análisis se mantiene
en st.session_state para que los módulos de cada página accedan a él.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    APP_NAME, APP_DESCRIPTION, APP_VERSION,
    CSV_SEPARATORS, CSV_ENCODINGS, MAX_FILE_SIZE_MB,
    PREVIEW_ROWS, COLOR_PRIMARY, COLOR_ACCENT,
)
from src.data_loader import load_csv, load_excel, get_file_info
from src.data_profiler import quick_quality_summary
from src.history_manager import HistoryManager
from src.visualizations import quality_gauge

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"{APP_NAME} v{APP_VERSION} — Sistema de análisis de datos",
    },
)

# ─────────────────────────────────────────────
# CSS global
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2980B9 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(30,58,95,0.25);
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1.1rem;
    opacity: 0.88;
    margin: 0.5rem 0 0;
}

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border-left: 4px solid #1E3A5F;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1E3A5F;
    line-height: 1;
}
.kpi-card .kpi-label {
    font-size: 0.8rem;
    color: #6C757D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.kpi-card .kpi-icon { font-size: 1.4rem; margin-bottom: 6px; }

/* Alert cards */
.alert-success {
    background: #D5F5E3; border-left: 4px solid #2ECC71;
    padding: 12px 16px; border-radius: 8px; color: #1E8449;
}
.alert-warning {
    background: #FEF9E7; border-left: 4px solid #F39C12;
    padding: 12px 16px; border-radius: 8px; color: #7D6608;
}
.alert-error {
    background: #FADBD8; border-left: 4px solid #E74C3C;
    padding: 12px 16px; border-radius: 8px; color: #922B21;
}
.alert-info {
    background: #EBF5FB; border-left: 4px solid #2980B9;
    padding: 12px 16px; border-radius: 8px; color: #154360;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1E3A5F;
    border-bottom: 2px solid #2ECC71;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C4F73 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #2980B9 !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1E3A5F, #2980B9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(30,58,95,0.3) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Inicialización del estado global
# ─────────────────────────────────────────────
def init_state() -> None:
    """Inicializa las variables de estado de Streamlit."""
    defaults = {
        "df_original": None,        # DataFrame original (sólo lectura)
        "df_work": None,            # DataFrame de trabajo (modificable)
        "file_info": None,          # Metadatos del archivo
        "project_name": "",         # Nombre del proyecto
        "project_desc": "",         # Descripción del proyecto
        "quality_profile": None,    # Perfil de calidad completo
        "quick_summary": None,      # Resumen rápido de calidad
        "cleaning_history": [],     # Historial de operaciones
        "ml_result": None,          # Resultado del último modelo ML
        "ts_result": None,          # Resultado del análisis temporal
        "sales_mapping": None,      # Mapping de columnas de ventas
        "sales_df": None,           # DataFrame de ventas normalizado
        "sales_metrics": None,      # Métricas de ventas
        "history_manager": HistoryManager(),
        "session_active": False,    # Indica si hay un análisis activo
        "analysis_summary": [],     # Resumen de análisis ejecutados
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ─────────────────────────────────────────────
# Barra lateral
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"# 📊 {APP_NAME}")
    st.markdown(f"*v{APP_VERSION}*")
    st.markdown("---")

    if st.session_state.session_active:
        st.markdown("### 📁 Proyecto activo")
        fi = st.session_state.file_info
        if fi:
            st.markdown(f"**{fi['nombre_archivo']}**")
            st.markdown(f"🔢 {fi['filas']:,} filas × {fi['columnas']} col.")
            st.markdown(f"💾 {fi['tamaño_mb']:.2f} MB")

        st.markdown("---")
        st.markdown("### 🗂️ Módulos")
        st.page_link("pages/1_Calidad_de_datos.py",     label="🔍 Calidad de datos")
        st.page_link("pages/2_Limpieza.py",              label="🧹 Limpieza")
        st.page_link("pages/3_Analisis_exploratorio.py", label="📈 EDA")
        st.page_link("pages/4_Analisis_de_ventas.py",   label="💰 Análisis de ventas")
        st.page_link("pages/5_Machine_Learning.py",      label="🤖 Machine Learning")
        st.page_link("pages/6_Historial.py",             label="📋 Historial")

        st.markdown("---")
        if st.button("🔄 Reiniciar análisis", use_container_width=True):
            # Limpiar estado sin borrar el HistoryManager
            hm = st.session_state.history_manager
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.history_manager = hm
            init_state()
            st.rerun()
    else:
        st.markdown("### 📖 Navegación")
        st.markdown("Carga un archivo para comenzar el análisis.")


# ─────────────────────────────────────────────
# Página principal
# ─────────────────────────────────────────────

if not st.session_state.session_active:
    # ── LANDING PAGE ──────────────────────────
    st.markdown("""
    <div class="main-header">
        <div style="font-size:3.5rem">📊</div>
        <h1>DataInsight AI</h1>
        <p>Sistema inteligente de análisis, procesamiento y predicción de datos</p>
        <p style="font-size:0.9rem;opacity:0.7;margin-top:8px">
            Carga tus archivos CSV o Excel y obtén estadísticas, visualizaciones y
            predicciones de forma automática
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Features destacadas
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🔍", "Calidad de datos", "Detecta valores faltantes, duplicados y anomalías"),
        ("🧹", "Limpieza", "Transforma y normaliza datos con historial de cambios"),
        ("📈", "Visualizaciones", "Gráficos interactivos adaptativos al tipo de dato"),
        ("🤖", "Machine Learning", "Modelos de regresión y clasificación con métricas"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        col.markdown(f"""
        <div class="kpi-card" style="text-align:center;border-left:4px solid #2ECC71">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-weight:600;color:#1E3A5F;margin:6px 0">{title}</div>
            <div style="font-size:0.8rem;color:#6C757D">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sample_file = ROOT / "data" / "samples" / "ventas_2023.csv"
    if sample_file.exists():
        st.markdown(
            """
            <div style="background:#f0f7ff;border:1px dashed #3498db;border-radius:8px;padding:12px 18px;margin-bottom:14px">
                <strong style="color:#1e3a5f">💡 ¿Deseas probar rápidamente sin subir un archivo?</strong><br>
                <span style="font-size:0.85rem;color:#555">Carga el dataset de prueba preinstalado: 365 ventas con fechas, categorías y métricas reales.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Cargar dataset de demostración (Ventas 2023)", type="secondary", use_container_width=True):
            df_sample = pd.read_csv(sample_file)
            file_info = {
                "nombre_archivo": "ventas_2023.csv",
                "tamaño_mb": round(sample_file.stat().st_size / (1024 * 1024), 3),
                "filas": len(df_sample),
                "columnas": len(df_sample.columns),
                "nombres_columnas": list(df_sample.columns),
                "tipos_originales": {c: str(df_sample[c].dtype) for c in df_sample.columns},
                "memoria_mb": round(df_sample.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            }
            quick = quick_quality_summary(df_sample)
            st.session_state.df_original = df_sample.copy()
            st.session_state.df_work = df_sample.copy()
            st.session_state.file_info = file_info
            st.session_state.project_name = "Demostración: Ventas 2023"
            st.session_state.project_desc = "Dataset sintético realista de ventas en Colombia"
            st.session_state.quick_summary = quick
            st.session_state.session_active = True
            st.session_state.cleaning_history = []
            st.session_state.history_manager.start_session(
                project="Demostración: Ventas 2023",
                file_name="ventas_2023.csv",
                rows_orig=df_sample.shape[0],
                cols_orig=df_sample.shape[1],
                quality=quick.get("porcentaje_faltantes", 0),
            )
            st.rerun()

    # ── FORMULARIO DE PROYECTO ──────────────────
    st.markdown('<div class="section-header">🚀 Comenzar nuevo análisis</div>', unsafe_allow_html=True)

    with st.form("project_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            project_name = st.text_input(
                "Nombre del proyecto *",
                placeholder="Ej: Análisis de ventas Q1 2024",
                help="Nombre descriptivo para identificar este análisis.",
            )
        with col_b:
            project_desc = st.text_input(
                "Descripción (opcional)",
                placeholder="Ej: Ventas por región y categoría de producto",
            )

        uploaded_file = st.file_uploader(
            "📂 Cargar archivo de datos *",
            type=["csv", "xlsx", "xls"],
            help=f"Archivos CSV o Excel. Tamaño máximo: {MAX_FILE_SIZE_MB} MB.",
        )

        # Opciones avanzadas de CSV (expandibles)
        with st.expander("⚙️ Opciones avanzadas (CSV)"):
            col_s, col_e = st.columns(2)
            with col_s:
                separator_label = st.selectbox(
                    "Separador",
                    options=list(CSV_SEPARATORS.keys()),
                    index=0,
                )
            with col_e:
                encoding = st.selectbox(
                    "Codificación",
                    options=CSV_ENCODINGS,
                    index=0,
                )

        # Para Excel: selección de hoja (se muestra después de cargar)
        submitted = st.form_submit_button(
            "🚀 Iniciar análisis",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not project_name.strip():
            st.markdown(
                '<div class="alert-error">❌ Por favor ingresa un nombre para el proyecto.</div>',
                unsafe_allow_html=True,
            )
        elif uploaded_file is None:
            st.markdown(
                '<div class="alert-error">❌ Por favor carga un archivo de datos.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("⏳ Cargando y validando el archivo…"):
                try:
                    sep = CSV_SEPARATORS[separator_label]
                    file_ext = uploaded_file.name.split(".")[-1].lower()

                    if file_ext == "csv":
                        df, warnings = load_csv(
                            uploaded_file, separator=sep, encoding=encoding
                        )
                        sheet_list = []
                    else:
                        # Para Excel primero detectamos las hojas
                        df, warnings, sheet_list = load_excel(uploaded_file)

                    file_info = get_file_info(uploaded_file, df)
                    quick = quick_quality_summary(df)

                    # Guardar en estado
                    st.session_state.df_original = df.copy()
                    st.session_state.df_work = df.copy()
                    st.session_state.file_info = file_info
                    st.session_state.project_name = project_name.strip()
                    st.session_state.project_desc = project_desc.strip()
                    st.session_state.quick_summary = quick
                    st.session_state.session_active = True
                    st.session_state.cleaning_history = []

                    # Iniciar sesión en el historial
                    st.session_state.history_manager.start_session(
                        project=project_name.strip(),
                        file_name=file_info["nombre_archivo"],
                        rows_orig=df.shape[0],
                        cols_orig=df.shape[1],
                        quality=quick.get("porcentaje_faltantes", 0),
                    )

                    # Advertencias
                    for w in warnings:
                        st.warning(w)

                    st.markdown(
                        f'<div class="alert-success">✅ Archivo cargado exitosamente: '
                        f'<strong>{file_info["nombre_archivo"]}</strong> — '
                        f'{file_info["filas"]:,} filas × {file_info["columnas"]} columnas.</div>',
                        unsafe_allow_html=True,
                    )
                    st.balloons()
                    st.rerun()

                except ValueError as e:
                    st.markdown(
                        f'<div class="alert-error">❌ Error al cargar el archivo: {e}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="alert-error">❌ Error inesperado: {e}</div>',
                        unsafe_allow_html=True,
                    )

else:
    # ── DASHBOARD PRINCIPAL ────────────────────
    fi = st.session_state.file_info
    qs = st.session_state.quick_summary
    df = st.session_state.df_work

    st.markdown(f"""
    <div class="main-header">
        <h1>📊 {st.session_state.project_name}</h1>
        <p>{st.session_state.project_desc or 'Análisis de datos interactivo'}</p>
        <p style="font-size:0.85rem;opacity:0.7">📁 {fi['nombre_archivo']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────
    st.markdown('<div class="section-header">📊 Resumen general</div>', unsafe_allow_html=True)

    kpi_cols = st.columns(6)
    kpis = [
        ("🔢", f"{qs['filas']:,}", "Registros"),
        ("📋", str(qs["columnas"]), "Variables"),
        ("❓", f"{qs['porcentaje_faltantes']:.1f}%", "Valores faltantes"),
        ("♻️", str(qs["filas_duplicadas"]), "Duplicados"),
        ("🔢", str(qs["columnas_numericas"]), "Col. numéricas"),
        ("🏷️", str(qs["columnas_categoricas"]), "Col. categóricas"),
    ]
    for col, (icon, value, label) in zip(kpi_cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Métricas de ventas (si existen) ────────
    if st.session_state.sales_metrics:
        sm = st.session_state.sales_metrics
        st.markdown('<div class="section-header">💰 Métricas de ventas</div>', unsafe_allow_html=True)
        v_cols = st.columns(3)
        if "ingresos_totales" in sm:
            v_cols[0].metric("💵 Ingresos totales", f"${sm['ingresos_totales']:,.2f}")
        if "unidades_totales" in sm:
            v_cols[1].metric("📦 Unidades vendidas", f"{sm['unidades_totales']:,.0f}")
        if "precio_promedio" in sm:
            v_cols[2].metric("💲 Precio promedio", f"${sm['precio_promedio']:,.2f}")

    # ── Vista previa de datos ──────────────────
    col_prev, col_qual = st.columns([3, 1])

    with col_prev:
        st.markdown('<div class="section-header">👁️ Vista previa de los datos</div>', unsafe_allow_html=True)

        tab_head, tab_tail, tab_info = st.tabs(["Primeras filas", "Últimas filas", "Información de columnas"])

        with tab_head:
            n_rows = st.slider("Filas a mostrar", 5, min(50, len(df)), PREVIEW_ROWS, key="prev_head")
            st.dataframe(df.head(n_rows), use_container_width=True)

        with tab_tail:
            n_rows_tail = st.slider("Filas a mostrar", 5, min(50, len(df)), PREVIEW_ROWS, key="prev_tail")
            st.dataframe(df.tail(n_rows_tail), use_container_width=True)

        with tab_info:
            info_df = pd.DataFrame({
                "Columna": df.columns,
                "Tipo": [str(t) for t in df.dtypes],
                "No nulos": [int(df[c].count()) for c in df.columns],
                "Únicos": [int(df[c].nunique()) for c in df.columns],
                "Faltantes": [int(df[c].isna().sum()) for c in df.columns],
                "Falt. (%)": [round(df[c].isna().mean() * 100, 1) for c in df.columns],
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

    with col_qual:
        st.markdown('<div class="section-header">🎯 Calidad</div>', unsafe_allow_html=True)
        if st.session_state.quality_profile:
            score = st.session_state.quality_profile.get("quality_score", 0)
        else:
            # Estimación rápida
            miss_pct = qs.get("porcentaje_faltantes", 0)
            dup_pct = qs["filas_duplicadas"] / max(qs["filas"], 1) * 100
            score = max(0, 100 - miss_pct * 1.5 - dup_pct * 0.5)
            score = round(score, 1)

        st.plotly_chart(quality_gauge(score), use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="kpi-card" style="margin-top:8px;border-left-color:#2ECC71">
            <div class="kpi-icon">📋</div>
            <div class="kpi-value">{len(st.session_state.cleaning_history)}</div>
            <div class="kpi-label">Operaciones realizadas</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Acciones de exportación ────────────────
    st.markdown('<div class="section-header">⬇️ Exportar datos actuales</div>', unsafe_allow_html=True)

    exp_cols = st.columns(3)

    from src.report_generator import export_csv, export_excel, generate_html_report

    with exp_cols[0]:
        csv_bytes = export_csv(df)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_bytes,
            file_name=f"{st.session_state.project_name.replace(' ', '_')}_limpio.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_cols[1]:
        excel_bytes = export_excel(df)
        st.download_button(
            label="📥 Descargar Excel",
            data=excel_bytes,
            file_name=f"{st.session_state.project_name.replace(' ', '_')}_limpio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with exp_cols[2]:
        html_report = generate_html_report(
            project_name=st.session_state.project_name,
            file_name=fi["nombre_archivo"],
            original_shape=(st.session_state.df_original.shape
                            if st.session_state.df_original is not None
                            else df.shape),
            clean_shape=df.shape,
            quality_score=score,
            history=st.session_state.cleaning_history,
            ml_metrics=st.session_state.ml_result.get("metricas")
                       if st.session_state.ml_result else None,
            sales_metrics=st.session_state.sales_metrics,
        )
        st.download_button(
            label="📄 Descargar informe HTML",
            data=html_report.encode("utf-8"),
            file_name=f"{st.session_state.project_name.replace(' ', '_')}_informe.html",
            mime="text/html",
            use_container_width=True,
        )
