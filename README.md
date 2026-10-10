# DataInsight AI

Sistema inteligente de análisis, procesamiento y predicción de datos construido con Python y Streamlit.

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Site-2ECC71?logo=github&logoColor=white)](https://skywalk3rcode.github.io/Parcial_Ing.SoftwareII/)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deploy%20App-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/deploy?repository=Skywalk3rcode/Parcial_Ing.SoftwareII&branch=main&mainModule=app.py)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-86%20Passed-2ECC71.svg)](tests/)

🌐 **Sitio Web en GitHub Pages:** [https://skywalk3rcode.github.io/Parcial_Ing.SoftwareII/](https://skywalk3rcode.github.io/Parcial_Ing.SoftwareII/)  
🚀 **Despliegue directo en Streamlit Cloud:** [Haz clic aquí para desplegar](https://share.streamlit.io/deploy?repository=Skywalk3rcode/Parcial_Ing.SoftwareII&branch=main&mainModule=app.py)

---

## 🚀 Características principales

| Módulo | Descripción |
|--------|-------------|
| 📂 Carga de datos | CSV y Excel con detección automática de separador y codificación |
| 🔍 Calidad de datos | Valores faltantes, duplicados, outliers y tipos incorrectos |
| 🧹 Limpieza | 10 operaciones con historial, undo y comparación antes/después |
| 📈 EDA | Estadísticas por tipo de dato y visualizaciones interactivas |
| 💰 Ventas | Análisis automático de datasets comerciales con KPIs |
| 🤖 Machine Learning | Regresión, clasificación y series temporales |
| 📋 Historial | Persistencia SQLite y exportación de resultados |

---

## 📦 Requisitos del sistema

- Python 3.10 o superior
- pip

---

## 🛠️ Instalación

### Windows

```powershell
# 1. Clonar o descomprimir el proyecto
cd "INGENIERIA DE SOFTWARE II\datainsight-ai"

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

### Linux / macOS

```bash
cd datainsight-ai

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

## 🧪 Ejecutar pruebas

```bash
# Desde la carpeta raíz del proyecto
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📁 Estructura del proyecto

```
datainsight-ai/
├── app.py                          # Punto de entrada (landing + dashboard)
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── settings.py                 # Configuración global centralizada
├── src/
│   ├── data_loader.py              # Carga y validación de CSV/Excel
│   ├── data_profiler.py            # Perfilado y calidad de datos
│   ├── data_cleaner.py             # Limpieza y transformación
│   ├── exploratory_analysis.py     # EDA estadístico
│   ├── sales_analysis.py           # Análisis de ventas
│   ├── ml_pipeline.py              # Machine Learning (sklearn Pipelines)
│   ├── time_series.py              # Series temporales
│   ├── visualizations.py           # Gráficos Plotly
│   ├── report_generator.py         # Exportación HTML/Excel/JSON
│   └── history_manager.py          # Historial SQLite
├── pages/
│   ├── 1_Calidad_de_datos.py
│   ├── 2_Limpieza.py
│   ├── 3_Analisis_exploratorio.py
│   ├── 4_Analisis_de_ventas.py
│   ├── 5_Machine_Learning.py
│   └── 6_Historial.py
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_data_cleaner.py
│   ├── test_sales_analysis.py
│   └── test_ml_pipeline.py
└── data/
    └── samples/
        └── ventas_2023.csv         # Dataset de ejemplo (365 registros)
```

---

## 🗂️ Dataset de muestra

El archivo `data/samples/ventas_2023.csv` contiene 365 registros de ventas ficticias con las columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha | texto (fecha) | Fecha de la venta |
| producto | texto | Nombre del producto |
| categoria | texto | Categoría del producto |
| cantidad | entero | Unidades vendidas |
| precio_unitario | decimal | Precio por unidad |
| ingresos_total | decimal | Total de la venta |
| cliente | texto | Tipo de cliente |
| region | texto | Región geográfica |
| vendedor | texto | Nombre del vendedor |

---

## 🧩 Flujo de uso recomendado

1. **Cargar datos** → Usa el formulario en la página principal.
2. **Revisar calidad** → `🔍 Calidad de datos` detecta problemas automáticamente.
3. **Limpiar datos** → `🧹 Limpieza` aplica operaciones con historial.
4. **Explorar** → `📈 EDA` genera estadísticas y gráficos por tipo de variable.
5. **Análisis de ventas** → `💰 Análisis de ventas` para datos comerciales.
6. **Entrenar modelo** → `🤖 Machine Learning` para regresión o clasificación.
7. **Exportar** → `📋 Historial` para descargar CSV, Excel, HTML o JSON.

---

## ⚙️ Configuración avanzada

Edita `config/settings.py` para ajustar:

```python
MAX_FILE_SIZE_MB = 100      # Tamaño máximo de archivo
ML_MIN_ROWS = 50            # Mínimo de filas para ML
OUTLIER_IQR_FACTOR = 1.5    # Factor para detección de outliers
TS_DEFAULT_HORIZON = 30     # Horizonte de predicción temporal
```

---

## ⚠️ Limitaciones conocidas

- El análisis de series temporales requiere mínimo 10 períodos agrupados.
- Los modelos de ML están limitados a los algoritmos de scikit-learn incluidos.
- El informe PDF requiere instalación adicional de `weasyprint` (no incluido).
- Las predicciones son estimaciones estadísticas, no garantías futuras.

---

## 🔒 Seguridad

- Los archivos cargados **no se almacenan permanentemente** en el servidor.
- No se ejecutan macros ni fórmulas de Excel.
- El archivo original nunca se modifica; todas las operaciones se aplican sobre una copia.
- No incluyas credenciales ni datos sensibles en los archivos de configuración.

---

## 📚 Tecnologías utilizadas

| Tecnología | Versión mínima | Uso |
|------------|----------------|-----|
| Python | 3.10 | Lenguaje principal |
| Streamlit | 1.32 | Interfaz web |
| Pandas | 2.0 | Manipulación de datos |
| NumPy | 1.24 | Cálculo numérico |
| Plotly | 5.18 | Visualizaciones interactivas |
| Scikit-learn | 1.4 | Machine Learning |
| OpenPyXL | 3.1 | Lectura/escritura Excel |
| SciPy | 1.11 | Estadística avanzada |
| SQLite | (stdlib) | Historial persistente |
| Pytest | 7.4 | Pruebas automatizadas |

---

## 👥 Autoría

Proyecto académico — Ingeniería de Software II  
DataInsight AI v1.0.0
