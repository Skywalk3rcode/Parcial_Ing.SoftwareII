"""
3_Analisis_exploratorio.py
--------------------------
Módulo de Análisis Exploratorio de Datos (EDA) con visualizaciones interactivas.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME
from src.exploratory_analysis import (
    numeric_stats, categorical_stats, temporal_stats,
    all_numeric_stats, all_categorical_stats,
    correlation_matrix, top_correlations, time_series_agg,
)
from src.visualizations import (
    histogram, boxplot, bar_chart, line_chart,
    scatter_plot, correlation_heatmap, pie_chart,
)

st.set_page_config(
    page_title=f"EDA — {APP_NAME}",
    page_icon="📈",
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
.stat-card {
    background: white; border-radius: 10px; padding: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #DEE2E6;
    text-align: center;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: #1E3A5F; }
.stat-lbl { font-size: 0.75rem; color: #6C757D; text-transform: uppercase; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A5F 0%, #2C4F73 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("session_active", False):
    st.warning("⚠️ No hay ningún archivo cargado.")
    st.stop()

df = st.session_state.df_work

st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2980B9);padding:1.5rem 2rem;
border-radius:12px;color:white;margin-bottom:1.5rem">
    <h2 style="margin:0">📈 Análisis Exploratorio de Datos</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Estadísticas descriptivas y visualizaciones interactivas por tipo de variable
    </p>
</div>
""", unsafe_allow_html=True)

# Clasificar columnas
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

# Resumen rápido
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total registros", f"{len(df):,}")
c2.metric("Variables numéricas", len(num_cols))
c3.metric("Variables categóricas", len(cat_cols))
c4.metric("Variables temporales", len(date_cols))

# ─────────────────────────────────────────────
# Tabs por tipo de variable
# ─────────────────────────────────────────────
eda_tabs = st.tabs([
    "🔢 Numéricas",
    "🏷️ Categóricas",
    "📅 Temporales",
    "🔗 Correlaciones",
    "📊 Comparar variables",
])

# ── Tab 1: Numéricas ──────────────────────────
with eda_tabs[0]:
    if not num_cols:
        st.info("No hay columnas numéricas en el dataset actual.")
    else:
        # Tabla general de estadísticas
        st.markdown('<div class="section-header">📋 Estadísticas descriptivas globales</div>', unsafe_allow_html=True)
        stats_df = all_numeric_stats(df)
        if not stats_df.empty:
            st.dataframe(stats_df.round(4), use_container_width=True)

        st.markdown('<div class="section-header">🔍 Análisis por columna</div>', unsafe_allow_html=True)
        sel_num = st.selectbox("Selecciona una variable numérica", num_cols, key="sel_num")

        if sel_num:
            stats = numeric_stats(df, sel_num)
            if "error" not in stats:
                # KPIs
                r1 = st.columns(4)
                r2 = st.columns(4)
                kpi_items = [
                    ("Media", f"{stats['media']:.4f}"),
                    ("Mediana", f"{stats['mediana']:.4f}"),
                    ("Moda", f"{stats['moda']:.4f}" if stats['moda'] is not None else "N/A"),
                    ("Desv. estándar", f"{stats['std']:.4f}"),
                    ("Mínimo", f"{stats['min']:.4f}"),
                    ("Máximo", f"{stats['max']:.4f}"),
                    ("Q1 (25%)", f"{stats['q1']:.4f}"),
                    ("Q3 (75%)", f"{stats['q3']:.4f}"),
                ]
                for col, (lbl, val) in zip(r1 + r2, kpi_items):
                    col.markdown(f"""<div class="stat-card">
                        <div class="stat-val">{val}</div>
                        <div class="stat-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_hist, col_box = st.columns(2)
                with col_hist:
                    bins = st.slider("Número de barras (histograma)", 5, 100, 30, key=f"bins_{sel_num}")
                    st.plotly_chart(histogram(df, sel_num, bins=bins), use_container_width=True)
                with col_box:
                    group_by_opt = st.selectbox(
                        "Agrupar boxplot por (opcional)",
                        ["(ninguno)"] + cat_cols,
                        key=f"box_grp_{sel_num}",
                    )
                    grp = group_by_opt if group_by_opt != "(ninguno)" else None
                    st.plotly_chart(boxplot(df, sel_num, group_by=grp), use_container_width=True)

                # Más detalles
                with st.expander("📊 Más estadísticas"):
                    detail_cols = st.columns(4)
                    detail_items = [
                        ("IQR", f"{stats['iqr']:.4f}"),
                        ("Varianza", f"{stats['varianza']:.4f}"),
                        ("Asimetría", f"{stats['asimetria']:.4f}"),
                        ("Curtosis", f"{stats['curtosis']:.4f}"),
                    ]
                    for col, (lbl, val) in zip(detail_cols, detail_items):
                        col.markdown(f"""<div class="stat-card">
                            <div class="stat-val">{val}</div>
                            <div class="stat-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

# ── Tab 2: Categóricas ────────────────────────
with eda_tabs[1]:
    if not cat_cols:
        st.info("No hay columnas categóricas en el dataset actual.")
    else:
        sel_cat = st.selectbox("Selecciona una variable categórica", cat_cols, key="sel_cat")
        top_n = st.slider("Top categorías a mostrar", 5, 50, 20, key="top_n_cat")

        if sel_cat:
            stats = categorical_stats(df, sel_cat, top_n=top_n)
            if "error" not in stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total registros", stats["total"])
                c2.metric("Categorías únicas", stats["unicos"])
                c3.metric("Categoría más frecuente", str(stats["categoria_mas_frecuente"]))
                c4.metric("Frecuencia máxima", stats["frecuencia_maxima"])

                col_bar, col_pie = st.columns(2)
                with col_bar:
                    st.plotly_chart(
                        bar_chart(df, sel_cat, top_n=top_n, horizontal=True),
                        use_container_width=True,
                    )
                with col_pie:
                    labels = list(stats["top_categorias"].keys())
                    values = list(stats["top_categorias"].values())
                    st.plotly_chart(
                        pie_chart(labels, values, title=f"Distribución de '{sel_cat}'"),
                        use_container_width=True,
                    )

                st.markdown('<div class="section-header">📊 Tabla de frecuencias</div>', unsafe_allow_html=True)
                freq_df = pd.DataFrame({
                    "Categoría": list(stats["top_categorias"].keys()),
                    "Frecuencia": list(stats["top_categorias"].values()),
                    "Porcentaje (%)": list(stats["top_porcentajes"].values()),
                })
                st.dataframe(freq_df, use_container_width=True, hide_index=True)

# ── Tab 3: Temporales ─────────────────────────
with eda_tabs[2]:
    all_date_cols = date_cols + [
        c for c in cat_cols
        if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.7
    ]

    if not all_date_cols:
        st.info("No se detectaron columnas temporales. Convierte columnas de fecha en el módulo de Limpieza.")
    else:
        sel_date = st.selectbox("Columna de fecha", all_date_cols, key="sel_date")
        sel_val = st.selectbox(
            "Variable a analizar en el tiempo",
            num_cols if num_cols else ["(sin variables numéricas)"],
            key="sel_ts_val",
        )
        freq_options = {"Día": "D", "Semana": "W", "Mes": "ME", "Trimestre": "QE", "Año": "YE"}
        freq_label = st.radio(
            "Frecuencia de agrupación",
            list(freq_options.keys()),
            index=2,
            horizontal=True,
            key="ts_freq",
        )
        agg_label = st.radio(
            "Función de agregación",
            ["sum", "mean", "count", "median"],
            index=0,
            horizontal=True,
            key="ts_agg",
        )

        if sel_val and sel_val != "(sin variables numéricas)":
            try:
                ts_df = time_series_agg(df, sel_date, sel_val, freq_options[freq_label], agg_label)
                if ts_df.empty:
                    st.warning("No hay datos suficientes para el período seleccionado.")
                else:
                    st.plotly_chart(
                        line_chart(ts_df, "fecha", sel_val,
                                   title=f"{sel_val} por {freq_label.lower()} ({agg_label})",
                                   add_trend=True),
                        use_container_width=True,
                    )
                    st.dataframe(ts_df.tail(24), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"❌ Error al generar la serie temporal: {e}")

# ── Tab 4: Correlaciones ──────────────────────
with eda_tabs[3]:
    if len(num_cols) < 2:
        st.info("Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
    else:
        corr_method = st.radio(
            "Método", ["pearson", "spearman", "kendall"],
            horizontal=True, key="corr_method"
        )
        corr = correlation_matrix(df, method=corr_method)

        if corr.empty:
            st.warning("No se pudo calcular la matriz de correlación.")
        else:
            st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

            st.markdown('<div class="section-header">🔝 Correlaciones más fuertes</div>', unsafe_allow_html=True)
            top_corr = top_correlations(df, method=corr_method, top_n=10)
            if not top_corr.empty:
                st.dataframe(top_corr, use_container_width=True, hide_index=True)

            st.info(
                "⚠️ **Nota:** La correlación indica relación estadística entre variables, "
                "no causalidad. Una correlación alta no implica que una variable cause la otra."
            )

# ── Tab 5: Comparar variables ─────────────────
with eda_tabs[4]:
    if len(num_cols) < 2:
        st.info("Se necesitan al menos 2 columnas numéricas.")
    else:
        col_x = st.selectbox("Variable X", num_cols, key="scat_x")
        col_y = st.selectbox("Variable Y", [c for c in num_cols if c != col_x], key="scat_y")
        col_color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, key="scat_color")

        color = col_color if col_color != "(ninguno)" else None
        st.plotly_chart(
            scatter_plot(df, col_x, col_y, color_col=color),
            use_container_width=True,
        )
