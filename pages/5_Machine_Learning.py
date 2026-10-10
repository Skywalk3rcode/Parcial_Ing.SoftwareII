"""
5_Machine_Learning.py
---------------------
Módulo de Machine Learning y predicciones (regresión, clasificación, series temporales).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME, ML_TEST_SIZE
from src.ml_pipeline import (
    check_ml_feasibility, train_regression, train_classification,
    predict_new, detect_task_type,
    REGRESSION_MODELS, CLASSIFICATION_MODELS,
)
from src.time_series import (
    check_ts_feasibility, prepare_time_series, train_time_series_model, forecast_future,
)
from src.visualizations import (
    actual_vs_predicted, confusion_matrix_chart, feature_importance_chart, line_chart,
)

st.set_page_config(
    page_title=f"Machine Learning — {APP_NAME}",
    page_icon="🤖",
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
.metric-card {
    background: linear-gradient(135deg, #1E3A5F, #2980B9);
    color: white; border-radius: 12px; padding: 1.25rem;
    text-align: center; box-shadow: 0 4px 15px rgba(30,58,95,0.3);
}
.metric-card .val { font-size: 2rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.8rem; opacity: 0.85; text-transform: uppercase; }
.metric-base { background: linear-gradient(135deg,#7F8C8D,#95A5A6) !important; }
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
    <h2 style="margin:0">🤖 Machine Learning y Predicciones</h2>
    <p style="margin:4px 0 0;opacity:0.85">
        Entrena modelos de regresión, clasificación o análisis de series temporales
    </p>
</div>
""", unsafe_allow_html=True)

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
all_cols = df.columns.tolist()
date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

ml_tabs = st.tabs([
    "🎯 Regresión / Clasificación",
    "📈 Series temporales",
    "🔮 Predicción de nuevos datos",
])

# ── Tab 1: Regresión / Clasificación ──────────
with ml_tabs[0]:
    st.markdown('<div class="section-header">⚙️ Configuración del modelo</div>', unsafe_allow_html=True)

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        target_col = st.selectbox(
            "🎯 Variable objetivo (a predecir)",
            all_cols,
            help="La columna que el modelo intentará predecir.",
            key="ml_target",
        )

        if target_col:
            auto_task = detect_task_type(df, target_col)
            st.info(f"Tipo de tarea detectado automáticamente: **{auto_task}**")

        task_type = st.radio(
            "Tipo de tarea",
            ["regresión", "clasificación"],
            horizontal=True,
            index=0 if auto_task == "regresión" else 1 if target_col else 0,
            key="ml_task",
        )

    with col_cfg2:
        feature_options = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "📋 Variables predictoras",
            feature_options,
            default=[c for c in num_cols if c != target_col][:5],
            key="ml_features",
            help="Columnas que el modelo usará para aprender.",
        )

    if task_type == "regresión":
        model_options = list(REGRESSION_MODELS.keys())
    else:
        model_options = list(CLASSIFICATION_MODELS.keys())

    model_name = st.selectbox("🧠 Modelo", model_options, key="ml_model")
    test_size = st.slider("Proporción de datos para prueba", 0.1, 0.4, ML_TEST_SIZE, 0.05, key="ml_test")

    # Verificar viabilidad
    if target_col and feature_cols:
        feasibility = check_ml_feasibility(df, target_col, feature_cols)
        if not feasibility["viable"]:
            st.error("❌ El conjunto de datos no es apto para entrenar un modelo:")
            for issue in feasibility["problemas"]:
                st.markdown(f"- {issue}")
        else:
            st.success(
                f"✅ El conjunto tiene {feasibility['n_filas']:,} filas y "
                f"{feasibility['n_features']} variables. Listo para entrenar."
            )

            if st.button("🚀 Entrenar modelo", key="btn_train", type="primary"):
                with st.spinner(f"⏳ Entrenando {model_name}…"):
                    try:
                        if task_type == "regresión":
                            result = train_regression(
                                df, target_col, feature_cols, model_name, test_size
                            )
                        else:
                            result = train_classification(
                                df, target_col, feature_cols, model_name, test_size
                            )

                        st.session_state.ml_result = result
                        st.session_state.history_manager.log_ml_result(
                            modelo=model_name,
                            tarea=task_type,
                            target=target_col,
                            metricas=result["metricas"],
                        )
                        st.success(f"✅ Modelo '{model_name}' entrenado exitosamente.")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento: {e}")

    # Mostrar resultados
    if st.session_state.get("ml_result"):
        result = st.session_state.ml_result
        metricas = result["metricas"]
        tarea = result["tarea"]

        st.markdown(f'<div class="section-header">📊 Resultados — {result["modelo"]}</div>', unsafe_allow_html=True)

        # KPIs de métricas
        metric_keys = {k: v for k, v in metricas.items() if "base" not in k.lower()}
        base_keys = {k: v for k, v in metricas.items() if "base" in k.lower()}

        metric_cols = st.columns(len(metric_keys))
        for col, (k, v) in zip(metric_cols, metric_keys.items()):
            col.markdown(f"""<div class="metric-card">
                <div class="val">{v:.4f}</div>
                <div class="lbl">{k}</div>
            </div>""", unsafe_allow_html=True)

        if base_keys:
            st.markdown("**📊 Comparación con modelo base (DummyRegressor/Classifier):**")
            base_cols = st.columns(len(base_keys))
            for col, (k, v) in zip(base_cols, base_keys.items()):
                col.markdown(f"""<div class="metric-card metric-base">
                    <div class="val">{v:.4f}</div>
                    <div class="lbl">{k}</div>
                </div>""", unsafe_allow_html=True)

        st.caption(
            f"Entrenamiento: {result['n_entrenamiento']:,} muestras | "
            f"Prueba: {result['n_prueba']:,} muestras"
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.plotly_chart(
                actual_vs_predicted(result["y_real"], result["y_pred"], tarea),
                use_container_width=True,
            )

        with col_right:
            if tarea == "clasificación" and "confusion_matrix" in result:
                st.plotly_chart(
                    confusion_matrix_chart(result["confusion_matrix"], result["clases"]),
                    use_container_width=True,
                )
            elif result.get("feature_importance"):
                st.plotly_chart(
                    feature_importance_chart(result["feature_importance"]),
                    use_container_width=True,
                )

        if tarea == "regresión":
            st.info(
                "ℹ️ **R² (coeficiente de determinación):** "
                "Valores cercanos a 1 indican mejor ajuste del modelo a los datos. "
                "Un R² bajo puede deberse a datos insuficientes o relaciones no lineales."
            )
        else:
            st.info(
                "ℹ️ **F1-Score:** Combina precisión y exhaustividad. "
                "Un valor cercano a 1 indica mejor desempeño del modelo."
            )

        st.warning(
            "⚠️ **Limitación:** Las métricas calculadas se basan en el conjunto de prueba disponible. "
            "El rendimiento real sobre datos nuevos puede variar. "
            "No interpretes estas predicciones como garantías futuras."
        )

# ── Tab 2: Series temporales ──────────────────
with ml_tabs[1]:
    st.markdown('<div class="section-header">⚙️ Configuración temporal</div>', unsafe_allow_html=True)

    all_date_candidates = date_cols + [
        c for c in cat_cols
        if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.5
    ]

    if not all_date_candidates:
        st.warning(
            "⚠️ No se detectaron columnas de fecha. "
            "Convierte una columna a datetime en el módulo de Limpieza."
        )
    elif not num_cols:
        st.warning("⚠️ No hay variables numéricas para analizar en el tiempo.")
    else:
        col_ts1, col_ts2 = st.columns(2)
        with col_ts1:
            ts_date_col = st.selectbox("📅 Columna de fecha", all_date_candidates, key="ts_date")
            ts_freq_opts = {"Día": "D", "Semana": "W", "Mes": "ME"}
            ts_freq_lbl = st.radio("Frecuencia", list(ts_freq_opts.keys()), index=2, horizontal=True, key="ts_freq_ml")
        with col_ts2:
            ts_val_col = st.selectbox("📊 Variable a predecir", num_cols, key="ts_val")
            ts_horizon = st.slider("Horizonte de predicción (períodos)", 3, 24, 6, key="ts_horizon")
            ts_model_name = st.selectbox("🧠 Modelo", ["Random Forest", "Regresión Lineal"], key="ts_model")

        ts_freq = ts_freq_opts[ts_freq_lbl]
        feasibility_ts = check_ts_feasibility(df, ts_date_col, ts_val_col, ts_freq)

        if not feasibility_ts["viable"]:
            for issue in feasibility_ts["problemas"]:
                st.error(f"❌ {issue}")
        else:
            st.success(
                f"✅ Serie con {feasibility_ts['n_periodos']} períodos "
                f"({feasibility_ts['fecha_inicio']} → {feasibility_ts['fecha_fin']})."
            )

            if st.button("🚀 Analizar y proyectar", key="btn_ts"):
                with st.spinner("⏳ Entrenando modelo temporal…"):
                    try:
                        ts_df = prepare_time_series(df, ts_date_col, ts_val_col, ts_freq)
                        ts_result = train_time_series_model(ts_df, ts_model_name, test_periods=min(6, len(ts_df)//4))
                        forecast_df = forecast_future(ts_result, horizon=ts_horizon, freq=ts_freq)
                        st.session_state.ts_result = {
                            "train_result": ts_result,
                            "forecast": forecast_df,
                            "ts_df": ts_df,
                        }
                        st.success("✅ Análisis temporal completado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Mostrar resultados temporales
        if st.session_state.get("ts_result"):
            ts_res = st.session_state.ts_result
            train_result = ts_res["train_result"]
            forecast_df = ts_res["forecast"]
            ts_df_hist = pd.DataFrame(ts_res["ts_df"])

            st.markdown('<div class="section-header">📈 Resultados temporales</div>', unsafe_allow_html=True)

            # Métricas
            m_cols = st.columns(4)
            metricas_ts = train_result["metricas"]
            for col, (k, v) in zip(m_cols, metricas_ts.items()):
                is_base = "referencia" in k.lower()
                col.markdown(f"""<div class="metric-card {'metric-base' if is_base else ''}">
                    <div class="val">{v:.4f}</div>
                    <div class="lbl">{k}</div>
                </div>""", unsafe_allow_html=True)

            # Gráfico histórico + proyección
            st.plotly_chart(
                line_chart(
                    ts_df_hist, "fecha", "valor",
                    title=f"Histórico y proyección — {ts_val_col}",
                    add_trend=False,
                    forecast_df=forecast_df,
                ),
                use_container_width=True,
            )

            # Tabla de proyección
            st.markdown("**📊 Proyección futura:**")
            forecast_display = forecast_df.copy()
            forecast_display["fecha"] = forecast_display["fecha"].dt.strftime("%Y-%m-%d")
            forecast_display.columns = ["Fecha", "Predicción"]
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)

            st.warning(
                "⚠️ **Limitaciones de la proyección:** Las predicciones son estimaciones "
                "basadas en patrones históricos. No tienen en cuenta eventos externos, "
                "cambios estacionales no observados, ni factores cualitativos. "
                "No deben interpretarse como certezas."
            )

# ── Tab 3: Predicción de nuevos datos ─────────
with ml_tabs[2]:
    ml_result = st.session_state.get("ml_result")
    if ml_result is None:
        st.info("ℹ️ Entrena un modelo primero en la pestaña 'Regresión / Clasificación'.")
    else:
        st.markdown(
            f'<div class="section-header">🔮 Predecir con {ml_result["modelo"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Variable objetivo:** `{ml_result['target_col']}` | "
            f"**Tarea:** {ml_result['tarea']}"
        )

        st.markdown("Introduce los valores de las variables predictoras:")
        new_data = {}
        feature_cols_ml = ml_result["feature_cols"]

        cols_pred = st.columns(min(3, len(feature_cols_ml)))
        for i, feat in enumerate(feature_cols_ml):
            col = cols_pred[i % len(cols_pred)]
            if pd.api.types.is_numeric_dtype(df[feat]):
                val = col.number_input(
                    feat,
                    value=float(df[feat].mean()) if df[feat].notna().sum() > 0 else 0.0,
                    key=f"pred_{feat}",
                )
            else:
                unique_vals = df[feat].dropna().unique().tolist()[:50]
                val = col.selectbox(feat, unique_vals, key=f"pred_{feat}")
            new_data[feat] = val

        if st.button("🔮 Generar predicción", key="btn_pred"):
            try:
                prediction = predict_new(ml_result, new_data)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1E3A5F,#2ECC71);
                color:white;padding:1.5rem 2rem;border-radius:12px;text-align:center;margin:1rem 0">
                    <div style="font-size:0.9rem;opacity:0.85;text-transform:uppercase">Predicción</div>
                    <div style="font-size:2.5rem;font-weight:700">{prediction}</div>
                    <div style="font-size:0.8rem;opacity:0.7;margin-top:6px">
                        {ml_result['target_col']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.warning(
                    "⚠️ Esta predicción es una estimación del modelo. "
                    "No garantiza el resultado real."
                )
            except Exception as e:
                st.error(f"❌ Error al generar la predicción: {e}")
