"""
test_data_loader.py
-------------------
Pruebas para el módulo de carga de archivos.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_csv, load_excel, validate_dataframe, get_file_info


class TestLoadCsv:
    def test_load_basic_csv(self):
        """Carga un CSV simple con separador coma."""
        csv_content = "nombre,edad,ciudad\nAna,25,Madrid\nBob,30,Lima\n"
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        file_obj.name = "test.csv"
        file_obj.size = len(csv_content)
        df, warnings = load_csv(file_obj, separator=",", encoding="utf-8")
        assert df.shape == (2, 3)
        assert list(df.columns) == ["nombre", "edad", "ciudad"]
        assert len(warnings) == 0

    def test_load_csv_semicolon_separator(self):
        """Carga un CSV con separador punto y coma."""
        csv_content = "nombre;edad\nAna;25\nBob;30\n"
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        file_obj.name = "test.csv"
        file_obj.size = len(csv_content)
        df, warnings = load_csv(file_obj, separator=";")
        assert df.shape == (2, 2)

    def test_empty_csv_raises(self):
        """Un CSV vacío debe lanzar ValueError."""
        csv_content = ""
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        file_obj.name = "empty.csv"
        file_obj.size = 0
        with pytest.raises(ValueError):
            load_csv(file_obj)

    def test_csv_only_headers_raises(self):
        """Un CSV sólo con encabezados debe lanzar ValueError."""
        csv_content = "col1,col2,col3\n"
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        file_obj.name = "headers_only.csv"
        file_obj.size = len(csv_content)
        with pytest.raises(ValueError):
            load_csv(file_obj)


class TestValidateDataframe:
    def test_valid_df_no_warnings(self, sample_df):
        """Un DataFrame válido no debe producir errores."""
        warnings = validate_dataframe(sample_df)
        assert isinstance(warnings, list)

    def test_empty_df_raises(self):
        """DataFrame vacío debe lanzar ValueError."""
        with pytest.raises(ValueError, match="vacío"):
            validate_dataframe(pd.DataFrame())

    def test_no_columns_raises(self):
        """DataFrame sin columnas debe lanzar ValueError."""
        df = pd.DataFrame(columns=[])
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_single_column_warning(self):
        """Un DataFrame con una sola columna debe generar advertencia."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        warnings = validate_dataframe(df)
        assert any("una columna" in w.lower() for w in warnings)


class TestGetFileInfo:
    def test_returns_expected_keys(self, sample_df):
        """get_file_info debe devolver las claves requeridas."""
        class MockFile:
            name = "test.csv"
            size = 1024

        info = get_file_info(MockFile(), sample_df)
        required_keys = [
            "nombre_archivo", "tamaño_mb", "filas", "columnas",
            "columnas_numericas", "columnas_categoricas",
        ]
        for key in required_keys:
            assert key in info

    def test_correct_dimensions(self, sample_df):
        """Las dimensiones deben coincidir con el DataFrame."""
        class MockFile:
            name = "ventas.csv"
            size = 5000

        info = get_file_info(MockFile(), sample_df)
        assert info["filas"] == len(sample_df)
        assert info["columnas"] == sample_df.shape[1]
