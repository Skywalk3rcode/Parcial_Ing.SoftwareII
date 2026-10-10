"""
test_ml_pipeline.py
-------------------
Pruebas para el módulo de Machine Learning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_pipeline import (
    check_ml_feasibility, train_regression, train_classification,
    predict_new, detect_task_type,
)


@pytest.fixture
def regression_df():
    """Dataset para regresión con 200 registros."""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    y = 3 * x1 + 2 * x2 + np.random.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": np.random.choice(["A", "B"], n), "y": y})


@pytest.fixture
def classification_df():
    """Dataset para clasificación binaria con 200 registros."""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    label = (x1 + x2 > 7).astype(str)
    label = np.where(label == "True", "Alto", "Bajo")
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


class TestCheckMlFeasibility:
    def test_feasible_dataset(self, regression_df):
        result = check_ml_feasibility(regression_df, "y", ["x1", "x2"])
        assert result["viable"]
        assert len(result["problemas"]) == 0

    def test_too_few_rows(self, small_df):
        result = check_ml_feasibility(small_df, "y", ["x"])
        assert not result["viable"]

    def test_missing_target_column(self, regression_df):
        result = check_ml_feasibility(regression_df, "columna_inexistente", ["x1"])
        assert not result["viable"]
        assert any("objetivo" in p.lower() or "no existe" in p.lower() for p in result["problemas"])


class TestTrainRegression:
    def test_returns_expected_keys(self, regression_df):
        result = train_regression(regression_df, "y", ["x1", "x2"], "Regresión Lineal")
        assert "metricas" in result
        assert "y_real" in result
        assert "y_pred" in result
        assert "pipeline" in result

    def test_metrics_have_expected_names(self, regression_df):
        result = train_regression(regression_df, "y", ["x1", "x2"])
        metricas = result["metricas"]
        assert "MAE" in metricas
        assert "RMSE" in metricas
        assert "R²" in metricas

    def test_r2_in_reasonable_range(self, regression_df):
        result = train_regression(regression_df, "y", ["x1", "x2"])
        assert result["metricas"]["R²"] > 0.5  # Datos linealmente separables

    def test_invalid_model_raises(self, regression_df):
        with pytest.raises(ValueError, match="no disponible"):
            train_regression(regression_df, "y", ["x1"], "ModeloInexistente")

    def test_with_categorical_features(self, regression_df):
        """El pipeline debe manejar variables categóricas automáticamente."""
        result = train_regression(regression_df, "y", ["x1", "x2", "cat"])
        assert result["metricas"]["R²"] is not None

    def test_random_forest_feature_importance(self, regression_df):
        result = train_regression(regression_df, "y", ["x1", "x2"], "Random Forest Regressor")
        assert "feature_importance" in result
        # Random Forest debe devolver importancias no vacías
        assert len(result["feature_importance"]) > 0


class TestTrainClassification:
    def test_returns_expected_keys(self, classification_df):
        result = train_classification(classification_df, "label", ["x1", "x2"])
        assert "metricas" in result
        assert "confusion_matrix" in result
        assert "clases" in result

    def test_metrics_have_expected_names(self, classification_df):
        result = train_classification(classification_df, "label", ["x1", "x2"])
        metricas = result["metricas"]
        assert "Accuracy" in metricas
        assert "F1-Score" in metricas

    def test_accuracy_in_valid_range(self, classification_df):
        result = train_classification(classification_df, "label", ["x1", "x2"])
        acc = result["metricas"]["Accuracy"]
        assert 0 <= acc <= 1

    def test_confusion_matrix_shape(self, classification_df):
        result = train_classification(classification_df, "label", ["x1", "x2"])
        cm = result["confusion_matrix"]
        n_classes = len(result["clases"])
        assert len(cm) == n_classes
        assert all(len(row) == n_classes for row in cm)


class TestPredictNew:
    def test_regression_prediction(self, regression_df):
        result = train_regression(regression_df, "y", ["x1", "x2"])
        pred = predict_new(result, {"x1": 5.0, "x2": 2.5})
        assert isinstance(pred, float)

    def test_classification_prediction_is_string(self, classification_df):
        result = train_classification(classification_df, "label", ["x1", "x2"])
        pred = predict_new(result, {"x1": 3.0, "x2": 1.0})
        assert pred in ["Alto", "Bajo"]

    def test_missing_pipeline_raises(self, regression_df):
        result = {"pipeline": None, "feature_cols": ["x1"], "tarea": "regresión"}
        with pytest.raises(ValueError, match="pipeline"):
            predict_new(result, {"x1": 5.0})


class TestDetectTaskType:
    def test_continuous_numeric_is_regression(self, regression_df):
        task = detect_task_type(regression_df, "y")
        assert task == "regresión"

    def test_string_target_is_classification(self, classification_df):
        task = detect_task_type(classification_df, "label")
        assert task == "clasificación"

    def test_few_unique_numeric_is_classification(self):
        df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
        task = detect_task_type(df, "y")
        assert task == "clasificación"
