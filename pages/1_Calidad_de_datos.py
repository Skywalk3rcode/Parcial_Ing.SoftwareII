"""
1_Calidad_de_datos.py
---------------------
Módulo de perfilado y calidad de datos.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME
from src.data_profiler import profile_dataset
from src.visualizations import missing_values_chart, quality_gauge

st.set_page_config(
    page_title=f"Calidad de datos — {APP_NAME}",
    page_icon="🔍",
    layout="wide",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #1E3A5F;
    border-bottom: 2px solid #2ECC71; padding-bottom: 6px; margin: 1.5rem 0 1rem;
}
.kpi-card {
    background: white; border-radius: 12px; padding: 1.25rem 1.5rem;
    border-left: 4px solid #1E3A5F; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1E3A5F; }
.kpi-label { font-size: 0.8rem; color: #6C757D; text-transform: uppercase; }
.badge-ok { background:#D5F5E3; color:#1E8449; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
.badge-warn { background:#FEF9E7; color:#7D6608; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
.badge-danger { background:#FADBD8; color:#922B21; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C4F73 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Verificar estado
# ─────────────────────────────────────────────
if not st.session_state.get("session_active", False):
    st.warning("⚠️ No hay ningún archivo cargado. Ve a la página principal para cargar un dataset.")
    st.stop()

df = st.session_state.df_work

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2980B9);padding:1.5rem 2rem;
border-radius:12px;color:white;margin-bottom:1.5rem">
    <h2 style="margin:0">🔍 Calidad de los datos</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Análisis automático de problemas y recomendaciones de mejora
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Ejecutar perfilado
# ─────────────────────────────────────────────
with st.spinner("⏳ Analizando la calidad del conjunto de datos…"):
    if st.session_state.get("quality_profile") is None:
        profile = profile_dataset(df)
        st.session_state.quality_profile = profile
    else:
        profile = st.session_state.quality_profile

# ─────────────────────────────────────────────
# Puntuación de calidad
# ─────────────────────────────────────────────
score = profile["quality_score"]
qs = st.session_state.quick_summary

col_gauge, col_kpis = st.columns([1, 3])

with col_gauge:
    st.plotly_chart(quality_gauge(score), use_container_width=True, config={"displayModeBar": False})

with col_kpis:
    st.markdown('<div class="section-header">📊 Indicadores de calidad</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{qs['valores_faltantes']:,}</div>
        <div class="kpi-label">Valores faltantes</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card" style="border-left-color:#E74C3C">
        <div class="kpi-value">{qs['filas_duplicadas']}</div>
        <div class="kpi-label">Filas duplicadas</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card" style="border-left-color:#F39C12">
        <div class="kpi-value">{len(profile['numeric_as_text'])}</div>
        <div class="kpi-label">Núm. como texto</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card" style="border-left-color:#8E44AD">
        <div class="kpi-value">{len(profile['dates_as_text'])}</div>
        <div class="kpi-label">Fechas como texto</div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs de análisis
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Tabla de calidad",
    "❓ Valores faltantes",
    "🔄 Duplicados",
    "⚠️ Valores extremos",
    "💡 Recomendaciones",
])

# Tab 1: Tabla de calidad general
with tab1:
    st.markdown('<div class="section-header">📋 Tabla de calidad por columna</div>', unsafe_allow_html=True)
    qt = profile["quality_table"]
    st.dataframe(qt, use_container_width=True, hide_index=True,
                 column_config={
                     "Faltantes (%)": st.column_config.ProgressColumn(
                         "Faltantes (%)", min_value=0, max_value=100
                     )
                 })

# Tab 2: Valores faltantes
with tab2:
    missing_df = profile["missing"]
    total_missing = int(missing_df["Faltantes"].sum())

    if total_missing == 0:
        st.success("✅ No hay valores faltantes en el conjunto de datos.")
    else:
        st.info(f"ℹ️ Se encontraron **{total_missing:,}** valores faltantes en total.")
        st.plotly_chart(
            missing_values_chart(missing_df), use_container_width=True
        )
        st.markdown("**Detalle por columna:**")
        st.dataframe(
            missing_df[missing_df["Faltantes"] > 0],
            use_container_width=True, hide_index=True
        )

# Tab 3: Duplicados
with tab3:
    dup_info = profile["duplicates"]
    if dup_info["cantidad"] == 0:
        st.success("✅ No hay filas duplicadas en el conjunto de datos.")
    else:
        st.warning(
            f"⚠️ Se encontraron **{dup_info['cantidad']}** filas duplicadas "
            f"({dup_info['porcentaje']:.2f}% del total)."
        )
        if st.checkbox("Mostrar filas duplicadas"):
            dup_rows = df[df.duplicated(keep=False)]
            st.dataframe(dup_rows.head(50), use_container_width=True)
            if len(dup_rows) > 50:
                st.caption(f"Mostrando 50 de {len(dup_rows)} filas duplicadas.")

# Tab 4: Valores extremos
with tab4:
    outliers = profile["outliers"]
    if not outliers:
        st.success("✅ No se detectaron valores extremos significativos.")
    else:
        st.warning(
            f"⚠️ Se detectaron valores extremos en **{len(outliers)}** columna(s). "
            "Esto no significa necesariamente que sean errores."
        )
        for col, info in outliers.items():
            with st.expander(f"📊 {col} — {info['iqr_count']} valores fuera del rango IQR"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Valores extremos (IQR)", info["iqr_count"])
                c2.metric("Valor mínimo extremo", f"{info['iqr_min']:.4f}")
                c3.metric("Valor máximo extremo", f"{info['iqr_max']:.4f}")

                series = df[col].dropna()
                st.plotly_chart(
                    __import__("src.visualizations", fromlist=["boxplot"]).boxplot(df, col),
                    use_container_width=True,
                )
        st.info(
            "💡 **Nota:** Los valores extremos pueden ser observaciones válidas. "
            "Analiza el contexto antes de eliminarlos. Puedes tratarlos en el módulo de **Limpieza**."
        )

# Tab 5: Recomendaciones
with tab5:
    st.markdown('<div class="section-header">💡 Recomendaciones automáticas</div>', unsafe_allow_html=True)

    suggestions = profile["dtype_suggestions"]
    format_issues = profile["format_issues"]
    id_cols = profile["possible_id_columns"]
    numeric_as_text = profile["numeric_as_text"]
    dates_as_text = profile["dates_as_text"]

    has_recommendations = any([suggestions, format_issues, id_cols, numeric_as_text, dates_as_text])

    if not has_recommendations:
        st.success("✅ No se encontraron problemas significativos. El conjunto de datos tiene buena estructura.")
    else:
        if numeric_as_text:
            st.markdown("**🔢 Columnas con números almacenados como texto:**")
            for col in numeric_as_text:
                st.markdown(f"- `{col}` → Convierte a tipo numérico en el módulo de **Limpieza**.")

        if dates_as_text:
            st.markdown("**📅 Columnas con fechas almacenadas como texto:**")
            for col in dates_as_text:
                st.markdown(f"- `{col}` → Convierte a tipo datetime en el módulo de **Limpieza**.")

        if suggestions:
            st.markdown("**⚙️ Conversiones de tipo sugeridas:**")
            sugg_df = pd.DataFrame(suggestions)
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)

        if format_issues:
            st.markdown("**✏️ Problemas de formato detectados:**")
            fmt_df = pd.DataFrame(format_issues)
            st.dataframe(fmt_df, use_container_width=True, hide_index=True)

        if id_cols:
            st.markdown("**🔑 Posibles columnas identificadoras (alta cardinalidad):**")
            for col in id_cols:
                st.markdown(
                    f"- `{col}` (únicos: {df[col].nunique()}) — "
                    "Considera excluirla de los modelos de ML."
                )

    st.info(
        "➡️ Para aplicar correcciones, ve al módulo **🧹 Limpieza** desde la barra lateral."
    )
