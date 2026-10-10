"""
report_generator.py
-------------------
Módulo para generar informes y exportar resultados.

Genera:
    - CSV del conjunto limpio.
    - Excel del conjunto limpio con múltiples hojas.
    - Informe HTML con estadísticas, historial y métricas.
    - Reporte de calidad en CSV.
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME, APP_VERSION


# ─────────────────────────────────────────────
# Exportar CSV
# ─────────────────────────────────────────────

def export_csv(df: pd.DataFrame, encoding: str = "utf-8-sig") -> bytes:
    """
    Exporta el DataFrame a CSV en formato bytes.
    Usa utf-8-sig para compatibilidad con Excel en Windows.
    """
    return df.to_csv(index=False, encoding=encoding).encode(encoding)


# ─────────────────────────────────────────────
# Exportar Excel
# ─────────────────────────────────────────────

def export_excel(
    df: pd.DataFrame,
    stats_df: Optional[pd.DataFrame] = None,
    quality_df: Optional[pd.DataFrame] = None,
    history: Optional[list[dict]] = None,
    sheet_name: str = "Datos Limpios",
) -> bytes:
    """
    Exporta el DataFrame a Excel con hojas adicionales opcionales.

    Parameters
    ----------
    df        : DataFrame principal (datos limpios).
    stats_df  : estadísticas descriptivas (opcional).
    quality_df: tabla de calidad (opcional).
    history   : historial de operaciones (opcional).
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])

        if stats_df is not None and not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="Estadísticas")

        if quality_df is not None and not quality_df.empty:
            quality_df.to_excel(writer, index=False, sheet_name="Calidad")

        if history:
            history_df = pd.DataFrame(history)
            history_df.to_excel(writer, index=False, sheet_name="Historial")

    return output.getvalue()


# ─────────────────────────────────────────────
# Informe HTML
# ─────────────────────────────────────────────

def generate_html_report(
    project_name: str,
    file_name: str,
    original_shape: tuple[int, int],
    clean_shape: tuple[int, int],
    quality_score: float,
    history: list[dict],
    numeric_stats: Optional[dict] = None,
    ml_metrics: Optional[dict] = None,
    sales_metrics: Optional[dict] = None,
) -> str:
    """
    Genera un informe HTML completo del análisis.

    Returns
    -------
    html_content : cadena HTML lista para descargar.
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Tabla de historial
    history_rows = ""
    for i, op in enumerate(history, 1):
        history_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{op.get('operacion', '')}</td>
            <td>{op.get('columna', '')}</td>
            <td>{op.get('afectados', '')}</td>
            <td>{op.get('descripcion', '')}</td>
        </tr>"""

    # Tabla de métricas ML
    ml_rows = ""
    if ml_metrics:
        for k, v in ml_metrics.items():
            ml_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    # Métricas de ventas
    sales_rows = ""
    if sales_metrics:
        for k, v in sales_metrics.items():
            sales_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    quality_color = (
        "#27AE60" if quality_score >= 75
        else "#F39C12" if quality_score >= 50
        else "#E74C3C"
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Análisis — {project_name}</title>
    <style>
        :root {{
            --primary: #1E3A5F;
            --accent: #2ECC71;
            --light: #F8F9FA;
            --border: #DEE2E6;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            margin: 0; padding: 0;
            background: var(--light);
            color: #2C3E50;
        }}
        .header {{
            background: linear-gradient(135deg, #1E3A5F, #2980B9);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 2rem; }}
        .header p {{ margin: 8px 0 0; opacity: 0.85; }}
        .container {{ max-width: 1000px; margin: 30px auto; padding: 0 20px; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            border: 1px solid var(--border);
        }}
        .card h2 {{
            color: var(--primary);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }}
        .metric {{
            background: var(--light);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border-left: 4px solid var(--primary);
        }}
        .metric .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .metric .label {{ font-size: 0.85rem; color: #7F8C8D; margin-top: 4px; }}
        .quality-badge {{
            display: inline-block;
            background: {quality_color};
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            padding: 8px 24px;
            border-radius: 30px;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{
            background: var(--primary);
            color: white;
            padding: 10px 14px;
            text-align: left;
        }}
        td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
        tr:hover {{ background: #F0F4F8; }}
        .footer {{
            text-align: center;
            color: #95A5A6;
            font-size: 0.8rem;
            padding: 20px;
        }}
        .note {{
            background: #EBF5FB;
            border-left: 4px solid #2980B9;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #154360;
            margin-top: 16px;
        }}
    </style>
</head>
<body>
<div class="header">
    <h1>📊 {APP_NAME}</h1>
    <p>Informe de Análisis — {project_name}</p>
    <p style="opacity:0.7;font-size:0.9rem">Generado: {now} | Versión {APP_VERSION}</p>
</div>

<div class="container">

    <!-- Información general -->
    <div class="card">
        <h2>📁 Información del Proyecto</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{file_name}</div>
                <div class="label">Archivo analizado</div>
            </div>
            <div class="metric">
                <div class="value">{original_shape[0]:,}</div>
                <div class="label">Filas originales</div>
            </div>
            <div class="metric">
                <div class="value">{original_shape[1]}</div>
                <div class="label">Columnas originales</div>
            </div>
            <div class="metric">
                <div class="value">{clean_shape[0]:,}</div>
                <div class="label">Filas tras limpieza</div>
            </div>
        </div>
    </div>

    <!-- Calidad -->
    <div class="card">
        <h2>🔍 Calidad de los Datos</h2>
        <p>Puntuación de calidad calculada automáticamente:</p>
        <div class="quality-badge">{quality_score}/100</div>
        <div class="note">
            ⚠️ La puntuación de calidad es orientativa. Un valor bajo indica presencia de
            datos faltantes, duplicados o tipos incorrectos que podrían afectar el análisis.
        </div>
    </div>

    <!-- Historial de operaciones -->
    <div class="card">
        <h2>🔧 Operaciones de Limpieza Realizadas</h2>
        {"<p>No se registraron operaciones de limpieza en esta sesión.</p>" if not history else f"""
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Operación</th>
                    <th>Columna</th>
                    <th>Afectados</th>
                    <th>Descripción</th>
                </tr>
            </thead>
            <tbody>{history_rows}</tbody>
        </table>"""}
    </div>

    {"" if not ml_metrics else f"""
    <!-- Resultados de Machine Learning -->
    <div class="card">
        <h2>🤖 Resultados de Machine Learning</h2>
        <table>
            <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
            <tbody>{ml_rows}</tbody>
        </table>
        <div class="note">
            ⚠️ Los resultados del modelo son estimaciones estadísticas. No garantizan
            predicciones exactas sobre datos futuros o no observados.
        </div>
    </div>"""}

    {"" if not sales_metrics else f"""
    <!-- Métricas de ventas -->
    <div class="card">
        <h2>💰 Métricas de Ventas</h2>
        <table>
            <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
            <tbody>{sales_rows}</tbody>
        </table>
    </div>"""}

</div>

<div class="footer">
    <p>Generado por {APP_NAME} v{APP_VERSION} | Informe académico — uso educativo</p>
</div>
</body>
</html>"""

    return html


# ─────────────────────────────────────────────
# Exportar JSON de configuración
# ─────────────────────────────────────────────

def export_config_json(
    project_name: str,
    file_name: str,
    history: list[dict],
    ml_config: Optional[dict] = None,
) -> str:
    """Exporta la configuración del análisis como JSON para reproducibilidad."""
    config = {
        "proyecto": project_name,
        "archivo": file_name,
        "fecha": datetime.now().isoformat(),
        "version": APP_VERSION,
        "operaciones": history,
        "ml_configuracion": ml_config or {},
    }
    return json.dumps(config, ensure_ascii=False, indent=2, default=str)
