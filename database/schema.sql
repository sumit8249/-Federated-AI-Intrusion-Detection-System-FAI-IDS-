-- ====================================================================
-- Federated AI Intrusion Detection System (FAI-IDS) Database Schema
-- Compatible with SQLite & PostgreSQL
-- Single Source of Truth: Section 14 of spec.md3.txt
-- ====================================================================

-- 1. Clients Table: Store organization/client details
CREATE TABLE IF NOT EXISTS Clients (
    client_id TEXT PRIMARY KEY,
    organization_name TEXT NOT NULL,
    organization_type TEXT NOT NULL,       -- e.g., Banking, Hospital, University, Tech Corp
    ip_address TEXT DEFAULT '127.0.0.1',
    status TEXT DEFAULT 'Active',          -- Active, Idle, Training, Disconnected
    dataset_records INTEGER DEFAULT 0,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. TrainingLogs Table: Accuracy, loss, training round per client
CREATE TABLE IF NOT EXISTS TrainingLogs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    client_id TEXT NOT NULL,
    train_loss REAL NOT NULL,
    train_accuracy REAL NOT NULL,
    val_loss REAL,
    val_accuracy REAL,
    samples_count INTEGER DEFAULT 0,
    duration_sec REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES Clients(client_id)
);

-- 3. AttackLogs Table: Detected attack information and intrusion telemetry
CREATE TABLE IF NOT EXISTS AttackLogs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    organization_name TEXT NOT NULL,
    attack_type TEXT NOT NULL,             -- Benign, DoS Hulk, DDoS, PortScan, etc.
    severity TEXT NOT NULL,                -- Critical, High, Medium, Low, Info
    confidence REAL NOT NULL,              -- 0.0 to 1.0
    source_ip TEXT,
    destination_port INTEGER,
    protocol TEXT,
    flow_duration REAL,
    total_packets INTEGER,
    total_bytes REAL,
    flow_summary TEXT,
    status TEXT DEFAULT 'Alerted',         -- Alerted, Mitigated, Quarantined, Investigating
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES Clients(client_id)
);

-- 4. GlobalModel Table: Store aggregated model versions and global metrics
CREATE TABLE IF NOT EXISTS GlobalModel (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    aggregated_accuracy REAL NOT NULL,
    aggregated_loss REAL NOT NULL,
    precision_score REAL DEFAULT 0.0,
    recall_score REAL DEFAULT 0.0,
    f1_score REAL DEFAULT 0.0,
    participating_clients INTEGER NOT NULL,
    weights_path TEXT,
    strategy_used TEXT DEFAULT 'FedAvg',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Predictions Table: Store prediction history for live flow inferences
CREATE TABLE IF NOT EXISTS Predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT,
    organization_name TEXT,
    predicted_class TEXT NOT NULL,
    probability REAL NOT NULL,
    is_intrusion INTEGER NOT NULL,         -- 1 if attack, 0 if benign
    severity TEXT,
    ground_truth TEXT,
    features_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for rapid dashboard queries
CREATE INDEX IF NOT EXISTS idx_training_round ON TrainingLogs(round_number);
CREATE INDEX IF NOT EXISTS idx_attack_detected ON AttackLogs(detected_at);
CREATE INDEX IF NOT EXISTS idx_attack_type ON AttackLogs(attack_type);
CREATE INDEX IF NOT EXISTS idx_global_round ON GlobalModel(round_number);
