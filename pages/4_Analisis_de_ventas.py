"""
4_Analisis_de_ventas.py
-----------------------
Módulo de análisis especializado para datos de ventas.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME
from src.sales_analysis import (
    detect_sales_columns, is_sales_dataset, prepare_sales_df,
    compute_sales_metrics, revenue_by_product, revenue_by_category,
    revenue_over_time, top_selling_products, activity_by_day,
)
from src.visualizations import (
    line_chart, bar_chart, pie_chart, sales_by_product_chart,
)
import plotly.express as px

st.set_page_config(
    page_title=f"Análisis de ventas — {APP_NAME}",
    page_icon="💰",
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
.kpi-card {
    background: white; border-radius: 12px; padding: 1.25rem 1.5rem;
    border-left: 4px solid #1E3A5F; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1E3A5F; }
.kpi-label { font-size: 0.8rem; color: #6C757D; text-transform: uppercase; }
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
    <h2 style="margin:0">💰 Análisis de Ventas</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Métricas, tendencias y comparativas para conjuntos de datos de ventas
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Detección automática de columnas
# ─────────────────────────────────────────────
auto_mapping = detect_sales_columns(df)
is_sales = is_sales_dataset(df, auto_mapping)

if not is_sales:
    st.warning(
        "⚠️ El dataset no parece contener datos de ventas, o las columnas requeridas no pudieron "
        "detectarse automáticamente. Puedes asignar las columnas manualmente a continuación."
    )

# ─────────────────────────────────────────────
# Asignación manual de columnas
# ─────────────────────────────────────────────
with st.expander("⚙️ Configurar columnas de ventas", expanded=not is_sales):
    st.markdown("Asigna las columnas del dataset a los roles de ventas. Deja en blanco si no aplica.")
    col_options = ["(no disponible)"] + df.columns.tolist()

    def _col_select(label, role, key_suffix):
        current = auto_mapping.get(role)
        default_idx = col_options.index(current) if current in col_options else 0
        return st.selectbox(label, col_options, index=default_idx, key=f"sales_{key_suffix}")

    c1, c2, c3 = st.columns(3)
    with c1:
        fecha_col    = _col_select("📅 Fecha de venta", "fecha", "fecha")
        cantidad_col = _col_select("📦 Cantidad", "cantidad", "cantidad")
        cliente_col  = _col_select("👤 Cliente", "cliente", "cliente")
    with c2:
        producto_col = _col_select("🏷️ Producto", "producto", "producto")
        precio_col   = _col_select("💲 Precio unitario", "precio", "precio")
        region_col   = _col_select("🌍 Región", "region", "region")
    with c3:
        cat_col      = _col_select("📂 Categoría", "categoria", "categoria")
        ingresos_col = _col_select("💵 Ingresos/Total", "ingresos", "ingresos")
        vendedor_col = _col_select("🧑‍💼 Vendedor", "vendedor", "vendedor")

    calc_revenue = False
    if ingresos_col == "(no disponible)" and cantidad_col != "(no disponible)" and precio_col != "(no disponible)":
        calc_revenue = st.checkbox(
            "✅ Calcular ingresos como Cantidad × Precio", value=True, key="calc_rev"
        )

    if st.button("🔄 Aplicar configuración de ventas", key="btn_sales_cfg"):
        st.session_state.sales_mapping = {
            "fecha":    fecha_col    if fecha_col    != "(no disponible)" else None,
            "producto": producto_col if producto_col != "(no disponible)" else None,
            "categoria":cat_col      if cat_col      != "(no disponible)" else None,
            "cantidad": cantidad_col if cantidad_col != "(no disponible)" else None,
            "precio":   precio_col   if precio_col   != "(no disponible)" else None,
            "ingresos": ingresos_col if ingresos_col != "(no disponible)" else None,
            "cliente":  cliente_col  if cliente_col  != "(no disponible)" else None,
            "region":   region_col   if region_col   != "(no disponible)" else None,
            "vendedor": vendedor_col if vendedor_col != "(no disponible)" else None,
        }
        try:
            sales_df = prepare_sales_df(df, st.session_state.sales_mapping, calculate_revenue=calc_revenue)
            st.session_state.sales_df = sales_df
            metrics = compute_sales_metrics(sales_df)
            st.session_state.sales_metrics = metrics
            st.success("✅ Configuración aplicada correctamente.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ {e}")

# Usar mapping guardado o el automático
if st.session_state.get("sales_mapping") is None:
    st.session_state.sales_mapping = auto_mapping

if st.session_state.get("sales_df") is None and is_sales:
    try:
        sales_df = prepare_sales_df(df, auto_mapping, calculate_revenue=True)
        st.session_state.sales_df = sales_df
        st.session_state.sales_metrics = compute_sales_metrics(sales_df)
    except Exception:
        pass

sales_df = st.session_state.get("sales_df")
metrics  = st.session_state.get("sales_metrics")
mapping  = st.session_state.get("sales_mapping", {})

if sales_df is None or sales_df.empty:
    st.info("ℹ️ Configura las columnas de ventas y aplica la configuración para ver el análisis.")
    st.stop()

# ─────────────────────────────────────────────
# KPIs de ventas
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Métricas generales</div>', unsafe_allow_html=True)

kpi_cols = st.columns(4)
kpi_data = []
if "ingresos_totales" in metrics:
    kpi_data.append(("💵", f"${metrics['ingresos_totales']:,.2f}", "Ingresos totales"))
if "unidades_totales" in metrics:
    kpi_data.append(("📦", f"{metrics['unidades_totales']:,.0f}", "Unidades vendidas"))
if "precio_promedio" in metrics:
    kpi_data.append(("💲", f"${metrics['precio_promedio']:,.2f}", "Precio promedio"))
if "total_registros" in metrics:
    kpi_data.append(("📋", f"{metrics['total_registros']:,}", "Total registros"))

for col, (icon, val, lbl) in zip(kpi_cols, kpi_data):
    col.markdown(f"""<div class="kpi-card">
        <div style="font-size:1.5rem">{icon}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

if "fecha_inicio" in metrics:
    st.caption(f"📅 Período: {metrics['fecha_inicio']} → {metrics['fecha_fin']} ({metrics.get('dias_activos', '?')} días)")

# ─────────────────────────────────────────────
# Análisis visual (tabs)
# ─────────────────────────────────────────────
sales_tabs = st.tabs([
    "📈 Ventas en el tiempo",
    "🏷️ Por producto",
    "📂 Por categoría",
    "🌍 Por región",
    "📅 Actividad por día",
])

# Tab: Tiempo
with sales_tabs[0]:
    if "fecha" not in sales_df.columns or "ingresos" not in sales_df.columns:
        st.info("Se necesitan columnas de fecha e ingresos para este análisis.")
    else:
        freq_opt = {"Día": "D", "Semana": "W", "Mes": "ME", "Trimestre": "QE"}
        freq_sel = st.radio(
            "Frecuencia", list(freq_opt.keys()), index=2, horizontal=True, key="freq_time"
        )
        val_col_ts = "ingresos" if "ingresos" in sales_df.columns else "cantidad"
        ts_df = revenue_over_time(sales_df, freq=freq_opt[freq_sel], value_col=val_col_ts)

        if not ts_df.empty:
            st.plotly_chart(
                line_chart(ts_df, "fecha", val_col_ts,
                           title=f"Evolución de {'ingresos' if val_col_ts == 'ingresos' else 'ventas'} — {freq_sel}",
                           add_trend=True),
                use_container_width=True,
            )
            st.dataframe(ts_df.tail(24), use_container_width=True, hide_index=True)

# Tab: Por producto
with sales_tabs[1]:
    prod_df = revenue_by_product(sales_df)
    if prod_df.empty:
        st.info("No se detectó columna de producto.")
    else:
        top_n_prod = st.slider("Top productos", 5, 30, 15, key="top_prod")
        st.plotly_chart(
            sales_by_product_chart(prod_df.head(top_n_prod)),
            use_container_width=True,
        )

        top_units = top_selling_products(sales_df, top_n=top_n_prod)
        if not top_units.empty:
            st.markdown('<div class="section-header">📦 Más vendidos por unidades</div>', unsafe_allow_html=True)
            st.dataframe(top_units, use_container_width=True, hide_index=True)

# Tab: Por categoría
with sales_tabs[2]:
    cat_df = revenue_by_category(sales_df)
    if cat_df.empty:
        st.info("No se detectó columna de categoría.")
    else:
        col_bar_cat, col_pie_cat = st.columns(2)
        with col_bar_cat:
            val_c = "ingresos" if "ingresos" in cat_df.columns else "cantidad"
            fig_cat = px.bar(
                cat_df.sort_values(val_c, ascending=True),
                x=val_c, y="categoria", orientation="h",
                title=f"{'Ingresos' if val_c == 'ingresos' else 'Unidades'} por categoría",
                color=val_c, color_continuous_scale=["#BDE3FF", "#1E3A5F"],
            )
            fig_cat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_cat, use_container_width=True)
        with col_pie_cat:
            if "participacion_pct" in cat_df.columns:
                st.plotly_chart(
                    pie_chart(
                        cat_df["categoria"].tolist(),
                        cat_df["ingresos"].tolist(),
                        title="Participación por categoría",
                    ),
                    use_container_width=True,
                )
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

# Tab: Por región
with sales_tabs[3]:
    if "region" not in sales_df.columns:
        st.info("No se detectó columna de región.")
    else:
        from src.sales_analysis import sales_by_region
        reg_df = sales_by_region(sales_df)
        if reg_df.empty:
            st.info("Sin datos de región disponibles.")
        else:
            val_r = "ingresos" if "ingresos" in reg_df.columns else "cantidad"
            fig_reg = px.bar(
                reg_df.sort_values(val_r, ascending=True),
                x=val_r, y="region", orientation="h",
                title=f"{'Ingresos' if val_r == 'ingresos' else 'Unidades'} por región",
                color=val_r, color_continuous_scale=["#BDE3FF", "#1E3A5F"],
            )
            fig_reg.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_reg, use_container_width=True)
            st.dataframe(reg_df, use_container_width=True, hide_index=True)

# Tab: Por día
with sales_tabs[4]:
    val_day = "ingresos" if "ingresos" in sales_df.columns else "cantidad"
    day_df = activity_by_day(sales_df, value_col=val_day)
    if day_df.empty:
        st.info("Se necesita columna de fecha para este análisis.")
    else:
        fig_day = px.bar(
            day_df, x="Día", y=val_day,
            title=f"{'Ingresos' if val_day == 'ingresos' else 'Unidades'} por día de la semana",
            color=val_day, color_continuous_scale=["#BDE3FF", "#1E3A5F"],
        )
        fig_day.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_day, use_container_width=True)
