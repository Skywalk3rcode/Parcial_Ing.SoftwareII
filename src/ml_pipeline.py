"""
ml_pipeline.py
--------------
Módulo de Machine Learning para regresión y clasificación.

Implementa pipelines de Scikit-learn que incluyen:
    - Preprocesamiento de variables numéricas y categóricas.
    - División entrenamiento/prueba sin fuga de información.
    - Entrenamiento de múltiples modelos.
    - Evaluación con métricas apropiadas.
    - Comparación contra modelos base (Dummy).
    - Predicción sobre nuevos registros.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    ML_TEST_SIZE,
    ML_RANDOM_STATE,
    ML_CV_FOLDS,
    ML_MIN_ROWS,
    ML_MIN_FEATURES,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Modelos disponibles
# ─────────────────────────────────────────────

REGRESSION_MODELS = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=ML_RANDOM_STATE
    ),
}

CLASSIFICATION_MODELS = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000, random_state=ML_RANDOM_STATE
    ),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100, random_state=ML_RANDOM_STATE
    ),
}


# ─────────────────────────────────────────────
# Validación de viabilidad
# ─────────────────────────────────────────────

def check_ml_feasibility(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Comprueba si el DataFrame es apto para entrenar un modelo.

    Returns
    -------
    result : dict con 'viable' (bool) y 'mensaje' (str) y diagnóstico adicional.
    """
    issues = []

    if len(df) < ML_MIN_ROWS:
        issues.append(
            f"El conjunto tiene {len(df)} filas. Se necesitan al menos {ML_MIN_ROWS}."
        )

    if target_col not in df.columns:
        issues.append(f"La columna objetivo '{target_col}' no existe.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        issues.append(f"Variables predictoras no encontradas: {missing_features}")

    if not issues:
        valid_features = [c for c in feature_cols if df[c].notna().sum() > 0]
        if len(valid_features) < ML_MIN_FEATURES:
            issues.append("No hay suficientes variables predictoras con datos válidos.")

        target_valid = df[target_col].dropna()
        if len(target_valid) < ML_MIN_ROWS * 0.5:
            issues.append(
                f"La variable objetivo tiene demasiados valores faltantes "
                f"({df[target_col].isna().sum()})."
            )

    return {
        "viable": len(issues) == 0,
        "problemas": issues,
        "n_filas": len(df),
        "n_features": len(feature_cols),
    }


# ─────────────────────────────────────────────
# Construcción del pipeline de preprocesamiento
# ─────────────────────────────────────────────

def _build_preprocessor(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> ColumnTransformer:
    """Construye un ColumnTransformer para variables numéricas y categóricas."""
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─────────────────────────────────────────────
# Entrenamiento y evaluación — Regresión
# ─────────────────────────────────────────────

def train_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str = "Random Forest Regressor",
    test_size: float = ML_TEST_SIZE,
) -> dict[str, Any]:
    """
    Entrena un modelo de regresión y retorna métricas y predicciones.
    """
    if model_name not in REGRESSION_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. "
            f"Opciones: {list(REGRESSION_MODELS.keys())}"
        )

    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = data[feature_cols]
    y = data[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=ML_RANDOM_STATE
    )

    preprocessor = _build_preprocessor(X_train, feature_cols)
    model = REGRESSION_MODELS[model_name]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Modelo base de comparación
    dummy = DummyRegressor(strategy="mean")
    dummy_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X_train, feature_cols)),
        ("model", dummy),
    ])
    dummy_pipeline.fit(X_train, y_train)
    y_pred_dummy = dummy_pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mae_dummy = float(mean_absolute_error(y_test, y_pred_dummy))

    # Importancia de características (solo Random Forest)
    feature_importance = {}
    try:
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            # Obtener nombres de features tras transformación
            ohe_cats = []
            prep = pipeline.named_steps["preprocessor"]
            if hasattr(prep, "named_transformers_"):
                if "cat" in prep.named_transformers_:
                    ohe_cats = list(
                        prep.named_transformers_["cat"].named_steps["encoder"]
                        .get_feature_names_out()
                    )
            num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
            all_feature_names = num_cols + ohe_cats
            importances = pipeline.named_steps["model"].feature_importances_
            for name, imp in zip(all_feature_names, importances):
                feature_importance[name] = round(float(imp), 4)
    except Exception:
        pass

    return {
        "modelo": model_name,
        "tarea": "regresión",
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "metricas": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
            "MAE_base": round(mae_dummy, 4),
        },
        "y_real": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "feature_importance": feature_importance,
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


# ─────────────────────────────────────────────
# Entrenamiento y evaluación — Clasificación
# ─────────────────────────────────────────────

def train_classification(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str = "Random Forest Classifier",
    test_size: float = ML_TEST_SIZE,
) -> dict[str, Any]:
    """
    Entrena un modelo de clasificación y retorna métricas y predicciones.
    """
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. "
            f"Opciones: {list(CLASSIFICATION_MODELS.keys())}"
        )

    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = data[feature_cols]

    # Codificar variable objetivo
    le = LabelEncoder()
    y = le.fit_transform(data[target_col].astype(str))
    class_names = list(le.classes_)

    # Verificar número mínimo de clases
    unique_classes = len(np.unique(y))
    if unique_classes < 2:
        raise ValueError(
            "La variable objetivo necesita al menos 2 categorías para clasificación."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=ML_RANDOM_STATE, stratify=y
        if min(np.bincount(y)) >= 2 else None
    )

    preprocessor = _build_preprocessor(X_train, feature_cols)
    model = CLASSIFICATION_MODELS[model_name]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Modelo base
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X_train, feature_cols)),
        ("model", dummy),
    ])
    dummy_pipeline.fit(X_train, y_train)
    y_pred_dummy = dummy_pipeline.predict(X_test)

    avg = "binary" if unique_classes == 2 else "weighted"

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    recall = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
    acc_dummy = float(accuracy_score(y_test, y_pred_dummy))

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Importancia de características
    feature_importance = {}
    try:
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            prep = pipeline.named_steps["preprocessor"]
            ohe_cats = []
            if hasattr(prep, "named_transformers_") and "cat" in prep.named_transformers_:
                ohe_cats = list(
                    prep.named_transformers_["cat"].named_steps["encoder"]
                    .get_feature_names_out()
                )
            num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
            all_feature_names = num_cols + ohe_cats
            importances = pipeline.named_steps["model"].feature_importances_
            for name, imp in zip(all_feature_names, importances):
                feature_importance[name] = round(float(imp), 4)
    except Exception:
        pass

    return {
        "modelo": model_name,
        "tarea": "clasificación",
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "clases": class_names,
        "metricas": {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Accuracy_base": round(acc_dummy, 4),
        },
        "y_real": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "pipeline": pipeline,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


# ─────────────────────────────────────────────
# Predicción sobre nuevos datos
# ─────────────────────────────────────────────

def predict_new(
    result: dict[str, Any],
    new_data: dict[str, Any],
) -> Any:
    """
    Genera una predicción para un nuevo registro.

    Parameters
    ----------
    result   : dict retornado por train_regression o train_classification.
    new_data : dict {nombre_columna: valor} con los valores del nuevo registro.
    """
    pipeline = result.get("pipeline")
    feature_cols = result.get("feature_cols", [])
    tarea = result.get("tarea")

    if pipeline is None:
        raise ValueError("El pipeline del modelo no está disponible.")

    # Construir DataFrame con el nuevo registro
    row = {col: new_data.get(col, np.nan) for col in feature_cols}
    df_new = pd.DataFrame([row])

    prediction = pipeline.predict(df_new)

    if tarea == "clasificación":
        le = result.get("label_encoder")
        if le is not None:
            prediction = le.inverse_transform(prediction)

    return prediction[0]


# ─────────────────────────────────────────────
# Detección automática del tipo de problema
# ─────────────────────────────────────────────

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detecta automáticamente si el problema es regresión o clasificación.

    Returns
    -------
    'regresión' o 'clasificación'.
    """
    if target_col not in df.columns:
        raise ValueError(f"La columna '{target_col}' no existe.")

    col = df[target_col].dropna()

    if pd.api.types.is_numeric_dtype(col):
        n_unique = col.nunique()
        if n_unique <= 10 and n_unique / len(col) < 0.05:
            return "clasificación"
        return "regresión"
    else:
        return "clasificación"
