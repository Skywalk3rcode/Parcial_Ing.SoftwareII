"""
6_Historial.py
--------------
Módulo de historial, reproducibilidad y exportación de resultados.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME
from src.report_generator import (
    export_csv, export_excel, generate_html_report, export_config_json,
)

st.set_page_config(
    page_title=f"Historial — {APP_NAME}",
    page_icon="📋",
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
.history-item {
    background: white; border-left: 4px solid #2980B9;
    padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.history-num {
    font-weight: 700; color: #1E3A5F; margin-right: 8px;
}
.history-desc { color: #5D6D7E; font-size: 0.9rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C4F73 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2980B9);padding:1.5rem 2rem;
border-radius:12px;color:white;margin-bottom:1.5rem">
    <h2 style="margin:0">📋 Historial y Exportación</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Registro de operaciones realizadas y opciones de exportación
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Historial de la sesión actual
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Operaciones de la sesión actual</div>', unsafe_allow_html=True)

history = st.session_state.get("cleaning_history", [])

if not history:
    st.info("ℹ️ No se han realizado operaciones de limpieza en esta sesión.")
else:
    for i, op in enumerate(history, 1):
        st.markdown(f"""
        <div class="history-item">
            <span class="history-num">#{i}</span>
            <strong>{op.get('operacion', '')}</strong>
            {f"— <em>{op.get('columna', '')}</em>" if op.get('columna') else ''}
            — Afectados: <strong>{op.get('afectados', 0)}</strong>
            <br><span class="history-desc">{op.get('descripcion', '')}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Resultados de ML
# ─────────────────────────────────────────────
ml_result = st.session_state.get("ml_result")
if ml_result:
    st.markdown('<div class="section-header">🤖 Último experimento de Machine Learning</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Modelo:** {ml_result['modelo']}")
        st.markdown(f"**Tarea:** {ml_result['tarea']}")
        st.markdown(f"**Variable objetivo:** `{ml_result['target_col']}`")
    with col2:
        st.markdown("**Métricas:**")
        for k, v in ml_result["metricas"].items():
            st.markdown(f"- {k}: **{v}**")

# ─────────────────────────────────────────────
# Historial de sesiones previas (SQLite)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Sesiones anteriores</div>', unsafe_allow_html=True)

hm = st.session_state.get("history_manager")
if hm:
    sessions_df = hm.get_all_sessions()
    if sessions_df.empty:
        st.info("No hay sesiones anteriores registradas en la base de datos.")
    else:
        st.dataframe(sessions_df, use_container_width=True, hide_index=True)

        # Detalle de una sesión específica
        if len(sessions_df) > 0:
            sel_session = st.selectbox(
                "Ver operaciones de una sesión",
                sessions_df["id"].tolist(),
                format_func=lambda x: f"Sesión #{x} — {sessions_df[sessions_df['id']==x]['project'].values[0] if len(sessions_df[sessions_df['id']==x]) > 0 else x}",
                key="hist_session",
            )
            if sel_session:
                ops_df = hm.get_session_operations(sel_session)
                if not ops_df.empty:
                    st.dataframe(ops_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay operaciones registradas para esta sesión.")

    # Resultados de ML históricos
    ml_df = hm.get_ml_results()
    if not ml_df.empty:
        st.markdown('<div class="section-header">📊 Experimentos de ML anteriores</div>', unsafe_allow_html=True)
        st.dataframe(ml_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Exportación
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">⬇️ Exportar resultados</div>', unsafe_allow_html=True)

session_active = st.session_state.get("session_active", False)

if not session_active:
    st.info("Carga un archivo para habilitar las opciones de exportación.")
else:
    df = st.session_state.df_work
    fi = st.session_state.file_info
    qs = st.session_state.quick_summary
    project_name = st.session_state.project_name

    col1, col2, col3, col4 = st.columns(4)

    # CSV
    with col1:
        csv_bytes = export_csv(df)
        st.download_button(
            label="📥 CSV limpio",
            data=csv_bytes,
            file_name=f"{project_name.replace(' ','_')}_datos_limpios.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Excel con múltiples hojas
    with col2:
        from src.exploratory_analysis import all_numeric_stats
        from src.data_profiler import profile_dataset

        try:
            stats_df = all_numeric_stats(df)
        except Exception:
            stats_df = pd.DataFrame()

        try:
            profile = st.session_state.get("quality_profile") or profile_dataset(df)
            quality_df = profile.get("quality_table", pd.DataFrame())
        except Exception:
            quality_df = pd.DataFrame()

        excel_bytes = export_excel(df, stats_df, quality_df, history)
        st.download_button(
            label="📥 Excel completo",
            data=excel_bytes,
            file_name=f"{project_name.replace(' ','_')}_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Informe HTML
    with col3:
        try:
            profile = st.session_state.get("quality_profile") or profile_dataset(df)
            quality_score = profile.get("quality_score", 0)
        except Exception:
            quality_score = 0

        html_report = generate_html_report(
            project_name=project_name,
            file_name=fi["nombre_archivo"],
            original_shape=(st.session_state.df_original.shape
                            if st.session_state.df_original is not None else df.shape),
            clean_shape=df.shape,
            quality_score=quality_score,
            history=history,
            ml_metrics=ml_result["metricas"] if ml_result else None,
            sales_metrics=st.session_state.get("sales_metrics"),
        )
        st.download_button(
            label="📄 Informe HTML",
            data=html_report.encode("utf-8"),
            file_name=f"{project_name.replace(' ','_')}_informe.html",
            mime="text/html",
            use_container_width=True,
        )

    # JSON de configuración
    with col4:
        ml_config = None
        if ml_result:
            ml_config = {
                "modelo": ml_result["modelo"],
                "tarea": ml_result["tarea"],
                "target": ml_result["target_col"],
                "features": ml_result["feature_cols"],
                "metricas": ml_result["metricas"],
            }

        config_json = export_config_json(
            project_name=project_name,
            file_name=fi["nombre_archivo"],
            history=history,
            ml_config=ml_config,
        )
        st.download_button(
            label="⚙️ Config JSON",
            data=config_json.encode("utf-8"),
            file_name=f"{project_name.replace(' ','_')}_config.json",
            mime="application/json",
            use_container_width=True,
        )

    # Predicciones de ML
    if ml_result:
        st.markdown('<div class="section-header">📊 Exportar predicciones del modelo</div>', unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            "Valor_real": ml_result["y_real"],
            "Valor_predicho": ml_result["y_pred"],
        })
        pred_csv = export_csv(pred_df)
        st.download_button(
            label="📥 Predicciones CSV",
            data=pred_csv,
            file_name=f"{project_name.replace(' ','_')}_predicciones.csv",
            mime="text/csv",
        )
