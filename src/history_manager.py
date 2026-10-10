"""
history_manager.py
------------------
Módulo para gestionar el historial de análisis con persistencia SQLite.

Registra:
    - Nombre del proyecto y archivo.
    - Dimensiones iniciales y finales.
    - Operaciones de limpieza.
    - Variables y modelos utilizados.
    - Métricas obtenidas.

Si SQLite no está disponible, mantiene el historial en memoria.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH


# ─────────────────────────────────────────────
# Inicialización de la base de datos
# ─────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Crea o abre la base de datos SQLite."""
    db_path = ROOT / DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db() -> bool:
    """
    Inicializa las tablas necesarias si no existen.
    Retorna True si la inicialización fue exitosa.
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project     TEXT NOT NULL,
            file_name   TEXT,
            created_at  TEXT,
            rows_orig   INTEGER,
            cols_orig   INTEGER,
            rows_clean  INTEGER,
            cols_clean  INTEGER,
            quality     REAL,
            notes       TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS operations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER REFERENCES sessions(id),
            step        INTEGER,
            operacion   TEXT,
            columna     TEXT,
            afectados   INTEGER,
            parametros  TEXT,
            descripcion TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER REFERENCES sessions(id),
            modelo      TEXT,
            tarea       TEXT,
            target      TEXT,
            metricas    TEXT,
            created_at  TEXT
        )
        """)

        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Clase principal del historial
# ─────────────────────────────────────────────

class HistoryManager:
    """
    Gestiona el historial de un análisis.
    Persiste en SQLite cuando está disponible; en caso contrario, usa memoria.
    """

    def __init__(self):
        self._db_available = init_db()
        self._session_id: Optional[int] = None
        self._operations: list[dict] = []  # Caché en memoria

    # ─────────────────────────────────────────
    # Sesión
    # ─────────────────────────────────────────

    def start_session(
        self,
        project: str,
        file_name: str,
        rows_orig: int,
        cols_orig: int,
        quality: float = 0.0,
        notes: str = "",
    ) -> int:
        """Inicia una nueva sesión de análisis."""
        self._operations = []
        now = datetime.now().isoformat()

        if self._db_available:
            try:
                conn = _get_connection()
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO sessions
                       (project, file_name, created_at, rows_orig, cols_orig, quality, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (project, file_name, now, rows_orig, cols_orig, quality, notes),
                )
                self._session_id = cur.lastrowid
                conn.commit()
                conn.close()
            except Exception:
                self._session_id = 0
        else:
            self._session_id = 0

        return self._session_id or 0

    def close_session(
        self,
        rows_clean: int,
        cols_clean: int,
        quality: float = 0.0,
    ) -> None:
        """Actualiza las dimensiones finales de la sesión."""
        if self._db_available and self._session_id:
            try:
                conn = _get_connection()
                cur = conn.cursor()
                cur.execute(
                    """UPDATE sessions
                       SET rows_clean=?, cols_clean=?, quality=?
                       WHERE id=?""",
                    (rows_clean, cols_clean, quality, self._session_id),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    # ─────────────────────────────────────────
    # Operaciones
    # ─────────────────────────────────────────

    def log_operation(self, operation: dict[str, Any]) -> None:
        """Registra una operación de limpieza."""
        self._operations.append(operation)

        if self._db_available and self._session_id:
            try:
                conn = _get_connection()
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO operations
                       (session_id, step, operacion, columna, afectados, parametros, descripcion)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self._session_id,
                        len(self._operations),
                        operation.get("operacion", ""),
                        str(operation.get("columna", "")),
                        operation.get("afectados", 0),
                        json.dumps(operation.get("parametros", {}), default=str),
                        operation.get("descripcion", ""),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    def log_ml_result(
        self,
        modelo: str,
        tarea: str,
        target: str,
        metricas: dict,
    ) -> None:
        """Registra el resultado de un experimento de Machine Learning."""
        if self._db_available and self._session_id:
            try:
                conn = _get_connection()
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO ml_results
                       (session_id, modelo, tarea, target, metricas, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        self._session_id,
                        modelo,
                        tarea,
                        target,
                        json.dumps(metricas, default=str),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    @property
    def operations(self) -> list[dict]:
        return list(self._operations)

    def clear_operations(self) -> None:
        self._operations = []

    # ─────────────────────────────────────────
    # Consulta de sesiones previas
    # ─────────────────────────────────────────

    def get_all_sessions(self) -> pd.DataFrame:
        """Devuelve todas las sesiones registradas en la base de datos."""
        if not self._db_available:
            return pd.DataFrame()
        try:
            conn = _get_connection()
            df = pd.read_sql_query(
                "SELECT * FROM sessions ORDER BY created_at DESC", conn
            )
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    def get_session_operations(self, session_id: int) -> pd.DataFrame:
        """Devuelve las operaciones de una sesión específica."""
        if not self._db_available:
            return pd.DataFrame()
        try:
            conn = _get_connection()
            df = pd.read_sql_query(
                "SELECT * FROM operations WHERE session_id=? ORDER BY step",
                conn, params=(session_id,)
            )
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    def get_ml_results(self) -> pd.DataFrame:
        """Devuelve todos los experimentos de ML registrados."""
        if not self._db_available:
            return pd.DataFrame()
        try:
            conn = _get_connection()
            df = pd.read_sql_query(
                "SELECT * FROM ml_results ORDER BY created_at DESC", conn
            )
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()
