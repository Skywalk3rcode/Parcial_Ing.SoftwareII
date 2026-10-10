"""
Configuración global de DataInsight AI.
Centraliza parámetros de la aplicación para facilitar su ajuste sin modificar el código de cada módulo.
"""

# ─────────────────────────────────────────────
# Información de la aplicación
# ─────────────────────────────────────────────
APP_NAME = "DataInsight AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Sistema inteligente de análisis, procesamiento y predicción de datos. "
    "Carga tus archivos CSV o Excel y obtén estadísticas, visualizaciones "
    "y predicciones de forma automática."
)

# ─────────────────────────────────────────────
# Límites de archivos
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB = 100          # Tamaño máximo en MB
MAX_ROWS_WARN = 500_000         # Filas a partir de las cuales se advierte al usuario
MAX_ROWS_HARD = 2_000_000       # Límite absoluto de filas (rendimiento)
PREVIEW_ROWS = 10               # Filas mostradas en la vista previa

# ─────────────────────────────────────────────
# Separadores CSV soportados
# ─────────────────────────────────────────────
CSV_SEPARATORS = {
    "Coma (,)": ",",
    "Punto y coma (;)": ";",
    "Tabulación (\\t)": "\t",
    "Barra vertical (|)": "|",
    "Espacio": " ",
}

# ─────────────────────────────────────────────
# Codificaciones soportadas
# ─────────────────────────────────────────────
CSV_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "latin-1",
    "iso-8859-1",
    "cp1252",
]

# ─────────────────────────────────────────────
# Umbrales de calidad de datos
# ─────────────────────────────────────────────
MISSING_WARN_THRESHOLD = 0.05   # >= 5 % → advertencia
MISSING_HIGH_THRESHOLD = 0.30   # >= 30 % → crítico
OUTLIER_IQR_FACTOR = 1.5        # Factor IQR para detección de valores extremos
OUTLIER_ZSCORE_THRESHOLD = 3.0  # Umbral Z-score alternativo

# ─────────────────────────────────────────────
# Machine Learning
# ─────────────────────────────────────────────
ML_TEST_SIZE = 0.2              # Proporción de datos para prueba
ML_RANDOM_STATE = 42            # Semilla de aleatoriedad
ML_CV_FOLDS = 5                 # Pliegues de validación cruzada
ML_MIN_ROWS = 50                # Mínimo de filas para entrenar un modelo
ML_MIN_FEATURES = 1             # Mínimo de variables predictoras

# ─────────────────────────────────────────────
# Series temporales
# ─────────────────────────────────────────────
TS_MIN_PERIODS = 10             # Mínimo de períodos para análisis temporal
TS_DEFAULT_HORIZON = 30         # Horizonte de predicción por defecto (días)

# ─────────────────────────────────────────────
# Historial (SQLite)
# ─────────────────────────────────────────────
DB_PATH = "data/history.db"

# ─────────────────────────────────────────────
# Paleta de colores principal
# ─────────────────────────────────────────────
COLOR_PRIMARY = "#1E3A5F"       # Azul oscuro
COLOR_ACCENT = "#2ECC71"        # Verde
COLOR_SECONDARY = "#3498DB"     # Azul medio
COLOR_WARNING = "#F39C12"       # Naranja
COLOR_DANGER = "#E74C3C"        # Rojo
COLOR_LIGHT = "#F8F9FA"         # Fondo claro
COLOR_CHART_SEQ = [
    "#1E3A5F", "#2980B9", "#1ABC9C", "#27AE60",
    "#8E44AD", "#E67E22", "#E74C3C", "#2ECC71",
]
