"""
time_series.py
--------------
Módulo de análisis y predicción de series temporales.

Implementa un flujo específico que respeta el orden cronológico en la validación,
sin mezclar datos futuros con el entrenamiento.

Características:
    - Agregación temporal configurable (día, semana, mes).
    - División cronológica train/test.
    - Predicción con modelos de regresión (features de ventana temporal).
    - Comparación con referencias sencillas (último valor, media histórica).
    - Horizonte de predicción configurable.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ML_RANDOM_STATE, TS_MIN_PERIODS, TS_DEFAULT_HORIZON


# ─────────────────────────────────────────────
# Preparación de la serie temporal
# ─────────────────────────────────────────────

def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Agrega los datos por período temporal y devuelve una serie limpia.

    Parameters
    ----------
    freq : 'D' (día), 'W' (semana), 'ME' (mes), 'QE' (trimestre).
    """
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna()
    temp = temp.set_index(date_col).sort_index()
    agg = temp[value_col].resample(freq).sum()
    agg = agg[agg.index.notnull()]
    result = agg.reset_index()
    result.columns = ["fecha", "valor"]
    return result


# ─────────────────────────────────────────────
# Ingeniería de features temporales
# ─────────────────────────────────────────────

def _create_lag_features(
    series: pd.Series,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Crea features de rezagos y ventanas deslizantes."""
    df_feat = pd.DataFrame({"valor": series.values})

    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["valor"].shift(lag)

    for window in rolling_windows:
        df_feat[f"rolling_mean_{window}"] = (
            df_feat["valor"].shift(1).rolling(window=window).mean()
        )
        df_feat[f"rolling_std_{window}"] = (
            df_feat["valor"].shift(1).rolling(window=window).std()
        )

    return df_feat.dropna()


# ─────────────────────────────────────────────
# Entrenamiento con división cronológica
# ─────────────────────────────────────────────

def train_time_series_model(
    ts_df: pd.DataFrame,
    model_name: str = "Random Forest",
    test_periods: int = 6,
    lags: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Entrena un modelo de series temporales con división cronológica.

    Parameters
    ----------
    ts_df       : DataFrame con columnas ['fecha', 'valor'].
    test_periods: Número de períodos al final usados para prueba.
    lags        : Lista de rezagos a usar como features.
    """
    if len(ts_df) < TS_MIN_PERIODS:
        raise ValueError(
            f"Se necesitan al menos {TS_MIN_PERIODS} períodos para el análisis temporal. "
            f"El conjunto tiene {len(ts_df)}."
        )

    if lags is None:
        lags = [1, 2, 3]

    rolling_windows = [3] if len(ts_df) >= 4 else []

    feat_df = _create_lag_features(ts_df["valor"], lags=lags, rolling_windows=rolling_windows)
    feat_dates = ts_df["fecha"].iloc[len(ts_df) - len(feat_df):]

    feature_cols = [c for c in feat_df.columns if c != "valor"]
    X = feat_df[feature_cols].values
    y = feat_df["valor"].values
    dates = feat_dates.values

    # División cronológica (NO aleatoria)
    if test_periods >= len(X):
        test_periods = max(1, len(X) // 4)

    split_idx = len(X) - test_periods
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    # Modelos
    models = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=ML_RANDOM_STATE
        ),
    }
    model = models.get(model_name, models["Random Forest"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Referencia sencilla: último valor conocido
    last_known = y_train[-1] if len(y_train) > 0 else 0.0
    y_pred_naive = np.full(len(y_test), last_known)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0
    mae_naive = float(mean_absolute_error(y_test, y_pred_naive))

    return {
        "modelo": model_name,
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "metricas": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
            "MAE_referencia": round(mae_naive, 4),
        },
        "y_real": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "fechas_prueba": [str(d)[:10] for d in dates_test],
        "historico": ts_df.to_dict(orient="records"),
        "model_obj": model,
        "feature_cols": feature_cols,
        "lags": lags,
        "last_known_values": y[-max(lags):].tolist(),
    }


# ─────────────────────────────────────────────
# Proyección futura
# ─────────────────────────────────────────────

def forecast_future(
    train_result: dict[str, Any],
    horizon: int = TS_DEFAULT_HORIZON,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Genera predicciones futuras usando el modelo entrenado.

    Parameters
    ----------
    horizon : número de períodos a proyectar.
    freq    : frecuencia temporal ('D', 'W', 'ME').

    Returns
    -------
    forecast_df : DataFrame con columnas ['fecha', 'prediccion'].

    Notes
    -----
    Las predicciones son estimaciones basadas en patrones históricos.
    No garantizan resultados futuros reales.
    """
    model = train_result.get("model_obj")
    lags = train_result.get("lags", [1, 2, 3])
    last_values = list(train_result.get("last_known_values", []))

    if model is None:
        raise ValueError("El modelo entrenado no está disponible para proyección.")

    # Determinar fecha de inicio de la proyección
    historico = pd.DataFrame(train_result.get("historico", []))
    if historico.empty:
        raise ValueError("No hay datos históricos para proyectar.")

    last_date = pd.to_datetime(historico["fecha"]).max()
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    predictions = []
    rolling_buffer = list(last_values)

    for _ in range(horizon):
        # Construir features con rezagos
        row_feats = []
        for lag in lags:
            idx = -lag
            val = rolling_buffer[idx] if len(rolling_buffer) >= abs(idx) else 0.0
            row_feats.append(val)

        # Features de ventana (si existen en el modelo)
        feature_cols = train_result.get("feature_cols", [])
        rolling_feats = [c for c in feature_cols if "rolling" in c]
        for _ in rolling_feats:
            window = 3
            win_vals = rolling_buffer[-window:] if len(rolling_buffer) >= window else rolling_buffer
            row_feats.append(np.mean(win_vals))
            row_feats.append(np.std(win_vals) if len(win_vals) > 1 else 0.0)

        X_future = np.array(row_feats[: len(feature_cols)]).reshape(1, -1)
        pred = float(model.predict(X_future)[0])
        pred = max(pred, 0)  # No predicciones negativas para valores como ventas
        predictions.append(pred)
        rolling_buffer.append(pred)

    forecast_df = pd.DataFrame({
        "fecha": future_dates,
        "prediccion": predictions,
    })
    return forecast_df


# ─────────────────────────────────────────────
# Validación de viabilidad temporal
# ─────────────────────────────────────────────

def check_ts_feasibility(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
) -> dict[str, Any]:
    """
    Comprueba si el conjunto es apto para análisis temporal.
    """
    issues = []

    if date_col not in df.columns:
        issues.append(f"La columna de fecha '{date_col}' no existe.")
        return {"viable": False, "problemas": issues}

    if value_col not in df.columns:
        issues.append(f"La columna objetivo '{value_col}' no existe.")
        return {"viable": False, "problemas": issues}

    try:
        ts = prepare_time_series(df, date_col, value_col, freq)
    except Exception as exc:
        issues.append(f"Error al preparar la serie: {exc}")
        return {"viable": False, "problemas": issues}

    if len(ts) < TS_MIN_PERIODS:
        issues.append(
            f"La serie tiene {len(ts)} períodos. Se necesitan al menos {TS_MIN_PERIODS}."
        )

    return {
        "viable": len(issues) == 0,
        "problemas": issues,
        "n_periodos": len(ts),
        "fecha_inicio": str(ts["fecha"].min())[:10] if len(ts) > 0 else None,
        "fecha_fin": str(ts["fecha"].max())[:10] if len(ts) > 0 else None,
    }
