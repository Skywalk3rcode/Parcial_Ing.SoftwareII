"""
visualizations.py
-----------------
Módulo centralizado de gráficos interactivos con Plotly.

Todas las funciones devuelven objetos plotly.graph_objects.Figure
listos para renderizar con st.plotly_chart().
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import COLOR_CHART_SEQ, COLOR_PRIMARY, COLOR_ACCENT, COLOR_DANGER


# ─────────────────────────────────────────────
# Tema global
# ─────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Inter, system-ui, sans-serif"


def _base_layout(**kwargs) -> dict:
    """Retorna opciones de layout base aplicadas a todos los gráficos."""
    return {
        "template": PLOTLY_TEMPLATE,
        "font": {"family": FONT_FAMILY, "size": 13},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
        **kwargs,
    }


# ─────────────────────────────────────────────
# Histograma de distribución numérica
# ─────────────────────────────────────────────

def histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    show_kde: bool = True,
) -> go.Figure:
    """Histograma interactivo con densidad opcional para una columna numérica."""
    series = df[column].dropna()

    fig = px.histogram(
        series,
        x=column,
        nbins=bins,
        title=f"Distribución de '{column}'",
        color_discrete_sequence=[COLOR_PRIMARY],
    )

    if show_kde:
        # Añadir curva de densidad
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series)
        x_range = np.linspace(series.min(), series.max(), 200)
        y_kde = kde(x_range)
        # Escalar la densidad al histograma
        hist_height = series.count() * (series.max() - series.min()) / bins
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde * hist_height,
                mode="lines",
                name="Densidad",
                line={"color": COLOR_ACCENT, "width": 2.5},
            )
        )

    fig.update_layout(
        **_base_layout(),
        xaxis_title=column,
        yaxis_title="Frecuencia",
        showlegend=show_kde,
    )
    return fig


# ─────────────────────────────────────────────
# Diagrama de caja (boxplot)
# ─────────────────────────────────────────────

def boxplot(
    df: pd.DataFrame,
    column: str,
    group_by: Optional[str] = None,
) -> go.Figure:
    """Boxplot interactivo con posible agrupación por categoría."""
    if group_by and group_by in df.columns:
        fig = px.box(
            df,
            x=group_by,
            y=column,
            title=f"Distribución de '{column}' por '{group_by}'",
            color=group_by,
            color_discrete_sequence=COLOR_CHART_SEQ,
        )
    else:
        fig = px.box(
            df,
            y=column,
            title=f"Diagrama de caja: '{column}'",
            color_discrete_sequence=[COLOR_PRIMARY],
        )

    fig.update_layout(**_base_layout(), yaxis_title=column)
    return fig


# ─────────────────────────────────────────────
# Gráfico de barras para categorías
# ─────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
    horizontal: bool = False,
) -> go.Figure:
    """Gráfico de barras para frecuencias de una variable categórica."""
    freq = df[column].value_counts().head(top_n).reset_index()
    freq.columns = [column, "Frecuencia"]

    if horizontal:
        fig = px.bar(
            freq,
            x="Frecuencia",
            y=column,
            orientation="h",
            title=f"Frecuencias de '{column}' (top {top_n})",
            color="Frecuencia",
            color_continuous_scale=["#BDE3FF", COLOR_PRIMARY],
        )
    else:
        fig = px.bar(
            freq,
            x=column,
            y="Frecuencia",
            title=f"Frecuencias de '{column}' (top {top_n})",
            color="Frecuencia",
            color_continuous_scale=["#BDE3FF", COLOR_PRIMARY],
        )

    fig.update_layout(**_base_layout(), coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────
# Gráfico de líneas para series temporales
# ─────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    add_trend: bool = False,
    forecast_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Gráfico de líneas temporal con tendencia opcional y proyección futura."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines+markers",
        name="Histórico",
        line={"color": COLOR_PRIMARY, "width": 2},
        marker={"size": 5},
    ))

    if add_trend:
        # Añadir línea de tendencia
        x_num = np.arange(len(df))
        m, b = np.polyfit(x_num, df[y_col].fillna(0), 1)
        trend_y = m * x_num + b
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=trend_y,
            mode="lines",
            name="Tendencia",
            line={"color": COLOR_ACCENT, "dash": "dash", "width": 2},
        ))

    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["prediccion"],
            mode="lines+markers",
            name="Proyección",
            line={"color": COLOR_DANGER, "dash": "dot", "width": 2},
            marker={"size": 6, "symbol": "diamond"},
        ))

    fig.update_layout(
        **_base_layout(title=title or f"{y_col} en el tiempo"),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


# ─────────────────────────────────────────────
# Matriz de correlación
# ─────────────────────────────────────────────

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Mapa de calor de la matriz de correlación."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        showscale=True,
    ))

    fig.update_layout(
        **_base_layout(title="Matriz de Correlación"),
        height=max(400, 60 * len(corr_matrix)),
    )
    return fig


# ─────────────────────────────────────────────
# Gráfico de dispersión
# ─────────────────────────────────────────────

def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    """Gráfico de dispersión interactivo."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"{x_col} vs {y_col}",
        color_discrete_sequence=COLOR_CHART_SEQ,
        opacity=0.7,
        trendline="ols",
    )
    fig.update_layout(**_base_layout(), xaxis_title=x_col, yaxis_title=y_col)
    return fig


# ─────────────────────────────────────────────
# Gráfico de pastel/donut
# ─────────────────────────────────────────────

def pie_chart(
    labels: list,
    values: list,
    title: str = "",
    donut: bool = True,
) -> go.Figure:
    """Gráfico de pastel o donut."""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4 if donut else 0,
        marker={"colors": COLOR_CHART_SEQ},
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Total: %{value}<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(**_base_layout(title=title), showlegend=True)
    return fig


# ─────────────────────────────────────────────
# Gráfico de calidad de datos (missing values)
# ─────────────────────────────────────────────

def missing_values_chart(missing_df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras para valores faltantes por columna."""
    data = missing_df[missing_df["Faltantes"] > 0].copy()
    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            **_base_layout(title="✅ Sin valores faltantes"),
            annotations=[{
                "text": "No hay valores faltantes en el conjunto de datos",
                "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 16},
            }]
        )
        return fig

    data = data.sort_values("Porcentaje (%)", ascending=True)

    fig = px.bar(
        data,
        x="Porcentaje (%)",
        y="Columna",
        orientation="h",
        title="Valores Faltantes por Columna (%)",
        color="Porcentaje (%)",
        color_continuous_scale=["#FFF3CD", "#FFC107", COLOR_DANGER],
        text="Porcentaje (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**_base_layout(), coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────
# Ventas: ingresos por producto
# ─────────────────────────────────────────────

def sales_by_product_chart(product_df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras horizontales: ingresos por producto."""
    col = "ingresos" if "ingresos" in product_df.columns else "cantidad"
    fig = px.bar(
        product_df.sort_values(col),
        x=col,
        y="producto",
        orientation="h",
        title=f"{'Ingresos' if col == 'ingresos' else 'Unidades'} por Producto",
        color=col,
        color_continuous_scale=["#BDE3FF", COLOR_PRIMARY],
        text=col,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(**_base_layout(), coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────
# ML: Real vs Predicho
# ─────────────────────────────────────────────

def actual_vs_predicted(y_real: list, y_pred: list, tarea: str = "regresión") -> go.Figure:
    """Gráfico de dispersión: valores reales vs predichos."""
    fig = go.Figure()

    if tarea == "regresión":
        fig.add_trace(go.Scatter(
            x=y_real, y=y_pred,
            mode="markers",
            name="Predicciones",
            marker={"color": COLOR_PRIMARY, "opacity": 0.6, "size": 8},
        ))
        # Línea perfecta
        min_val = min(min(y_real), min(y_pred))
        max_val = max(max(y_real), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predicción perfecta",
            line={"color": COLOR_ACCENT, "dash": "dash", "width": 2},
        ))
        fig.update_layout(
            **_base_layout(title="Real vs Predicho"),
            xaxis_title="Valor Real",
            yaxis_title="Valor Predicho",
        )
    else:
        # Para clasificación: mostrar índices
        fig.add_trace(go.Scatter(
            x=list(range(len(y_real))),
            y=y_real,
            mode="markers",
            name="Real",
            marker={"color": COLOR_PRIMARY, "size": 8},
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode="markers",
            name="Predicho",
            marker={"color": COLOR_ACCENT, "size": 6, "symbol": "x"},
        ))
        fig.update_layout(
            **_base_layout(title="Clases Reales vs Predichas"),
            xaxis_title="Índice",
            yaxis_title="Clase",
        )

    return fig


# ─────────────────────────────────────────────
# ML: Matriz de confusión
# ─────────────────────────────────────────────

def confusion_matrix_chart(cm: list[list], class_names: list[str]) -> go.Figure:
    """Heatmap de la matriz de confusión."""
    cm_array = np.array(cm)
    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=class_names,
        y=class_names,
        colorscale=[[0, "#EBF5FB"], [1, COLOR_PRIMARY]],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        showscale=True,
    ))

    fig.update_layout(
        **_base_layout(title="Matriz de Confusión"),
        xaxis_title="Predicho",
        yaxis_title="Real",
        height=max(300, 80 * len(class_names)),
    )
    return fig


# ─────────────────────────────────────────────
# ML: Importancia de variables
# ─────────────────────────────────────────────

def feature_importance_chart(
    feature_importance: dict[str, float],
    top_n: int = 15,
) -> go.Figure:
    """Gráfico de barras de importancia de variables."""
    if not feature_importance:
        return go.Figure()

    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*items)

    fig = px.bar(
        x=values,
        y=names,
        orientation="h",
        title=f"Importancia de Variables (top {top_n})",
        color=values,
        color_continuous_scale=["#BDE3FF", COLOR_PRIMARY],
        labels={"x": "Importancia", "y": "Variable"},
    )
    fig.update_layout(**_base_layout(), coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────
# Dashboard: KPI Gauge
# ─────────────────────────────────────────────

def quality_gauge(score: float) -> go.Figure:
    """Indicador tipo gauge para la puntuación de calidad."""
    color = (
        COLOR_ACCENT if score >= 75
        else "#F39C12" if score >= 50
        else COLOR_DANGER
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Puntuación de Calidad", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#FADBD8"},
                {"range": [50, 75], "color": "#FDEBD0"},
                {"range": [75, 100], "color": "#D5F5E3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/100", "font": {"size": 28}},
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font={"family": FONT_FAMILY, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig
