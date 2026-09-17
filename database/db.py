"""
Database Interface Module for FAI-IDS
Provides thread-safe operations for SQLite database.
Spec Reference: Section 14 (Database Design)
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fai_ids.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection configured with dict rows and foreign keys."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the database schema if not already initialized."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with get_connection(db_path) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


# ----------------------------------------------------------------------
# 1. Clients Management
# ----------------------------------------------------------------------

def register_client(client_id: str, organization_name: str, organization_type: str,
                    ip_address: str = "127.0.0.1", dataset_records: int = 0) -> None:
    """Registers or updates a federated client organization."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO Clients (client_id, organization_name, organization_type, ip_address, status, dataset_records, last_seen)
            VALUES (?, ?, ?, ?, 'Active', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(client_id) DO UPDATE SET
                organization_name=excluded.organization_name,
                organization_type=excluded.organization_type,
                ip_address=excluded.ip_address,
                status='Active',
                dataset_records=excluded.dataset_records,
                last_seen=CURRENT_TIMESTAMP
        """, (client_id, organization_name, organization_type, ip_address, dataset_records))
        conn.commit()


def update_client_status(client_id: str, status: str) -> None:
    """Updates client operational status."""
    with get_connection() as conn:
        conn.execute("""
            UPDATE Clients SET status = ?, last_seen = CURRENT_TIMESTAMP WHERE client_id = ?
        """, (status, client_id))
        conn.commit()


def get_all_clients() -> List[Dict[str, Any]]:
    """Fetches all registered client organizations."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM Clients ORDER BY registered_at ASC")
        return [dict(row) for row in cursor.fetchall()]


# ----------------------------------------------------------------------
# 2. Training Logs Management
# ----------------------------------------------------------------------

def log_training_round(round_number: int, client_id: str, train_loss: float,
                       train_accuracy: float, val_loss: Optional[float] = None,
                       val_accuracy: Optional[float] = None, samples_count: int = 0,
                       duration_sec: float = 0.0) -> int:
    """Logs local training round metrics for a client."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO TrainingLogs (round_number, client_id, train_loss, train_accuracy, val_loss, val_accuracy, samples_count, duration_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (round_number, client_id, float(train_loss), float(train_accuracy),
              float(val_loss) if val_loss is not None else None,
              float(val_accuracy) if val_accuracy is not None else None,
              int(samples_count), float(duration_sec)))
        conn.commit()
        return cursor.lastrowid


def get_training_logs(client_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves training logs, optionally filtered by client_id."""
    with get_connection() as conn:
        if client_id:
            cursor = conn.execute("""
                SELECT t.*, c.organization_name 
                FROM TrainingLogs t
                JOIN Clients c ON t.client_id = c.client_id
                WHERE t.client_id = ?
                ORDER BY round_number ASC
            """, (client_id,))
        else:
            cursor = conn.execute("""
                SELECT t.*, c.organization_name 
                FROM TrainingLogs t
                JOIN Clients c ON t.client_id = c.client_id
                ORDER BY round_number ASC, client_id ASC
            """)
        return [dict(row) for row in cursor.fetchall()]


# ----------------------------------------------------------------------
# 3. Global Model Versioning
# ----------------------------------------------------------------------

def log_global_model(round_number: int, aggregated_accuracy: float,
                     aggregated_loss: float, precision_score: float = 0.0,
                     recall_score: float = 0.0, f1_score: float = 0.0,
                     participating_clients: int = 3, weights_path: str = "") -> int:
    """Records global aggregated model evaluation after a federated round."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO GlobalModel (round_number, aggregated_accuracy, aggregated_loss, 
                                     precision_score, recall_score, f1_score, 
                                     participating_clients, weights_path, strategy_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'FedAvg')
        """, (int(round_number), float(aggregated_accuracy), float(aggregated_loss),
              float(precision_score), float(recall_score), float(f1_score),
              int(participating_clients), weights_path))
        conn.commit()
        return cursor.lastrowid


def get_global_models() -> List[Dict[str, Any]]:
    """Retrieves all aggregated global model versions."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM GlobalModel ORDER BY round_number ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_latest_global_model() -> Optional[Dict[str, Any]]:
    """Gets the latest aggregated global model stats."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM GlobalModel ORDER BY round_number DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None


# ----------------------------------------------------------------------
# 4. Attack Logs & Intrusion Telemetry
# ----------------------------------------------------------------------

def log_attack(client_id: str, organization_name: str, attack_type: str,
               severity: str, confidence: float, source_ip: str = "192.168.1.105",
               destination_port: int = 80, protocol: str = "TCP",
               flow_duration: float = 0.0, total_packets: int = 1,
               total_bytes: float = 0.0, flow_summary: str = "",
               status: str = "Alerted") -> int:
    """Logs an intrusion detection alert."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO AttackLogs (client_id, organization_name, attack_type, severity, 
                                    confidence, source_ip, destination_port, protocol, 
                                    flow_duration, total_packets, total_bytes, flow_summary, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (client_id, organization_name, attack_type, severity, float(confidence),
              source_ip, int(destination_port), protocol, float(flow_duration),
              int(total_packets), float(total_bytes), flow_summary, status))
        conn.commit()
        return cursor.lastrowid


def get_attack_logs(limit: int = 100, attack_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves detected attack logs for the dashboard."""
    with get_connection() as conn:
        query = "SELECT * FROM AttackLogs"
        params: list = []
        if attack_type:
            query += " WHERE attack_type = ?"
            params.append(attack_type)
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)
        cursor = conn.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def get_attack_counts_by_type() -> Dict[str, int]:
    """Returns counts of detected attacks grouped by attack type."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT attack_type, COUNT(*) as count 
            FROM AttackLogs 
            GROUP BY attack_type 
            ORDER BY count DESC
        """)
        return {row["attack_type"]: row["count"] for row in cursor.fetchall()}


# ----------------------------------------------------------------------
# 5. Predictions Management
# ----------------------------------------------------------------------

def log_prediction(client_id: str, organization_name: str, predicted_class: str,
                   probability: float, is_intrusion: int, severity: str,
                   ground_truth: Optional[str] = None,
                   features_json: Optional[str] = None) -> int:
    """Logs a live flow prediction."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO Predictions (client_id, organization_name, predicted_class, 
                                     probability, is_intrusion, severity, ground_truth, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (client_id, organization_name, predicted_class, float(probability),
              int(is_intrusion), severity, ground_truth, features_json))
        conn.commit()
        return cursor.lastrowid


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent live predictions."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM Predictions ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


# Automatically initialize DB when module is imported
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
