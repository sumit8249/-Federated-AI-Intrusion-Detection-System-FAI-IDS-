# Federated AI Intrusion Detection System (FAI-IDS)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FL-yellow.svg)](https://flower.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-brightgreen.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

A privacy-preserving cybersecurity framework where multiple organizations (e.g., Banks, Hospitals, Universities) collaboratively train a state-of-the-art intrusion detection model without sharing their sensitive raw network traffic data.

Built strictly in accordance with **[spec.md3.txt](spec.md3.txt)**.

---

## 1. Architecture Overview

Traditional Intrusion Detection Systems require centralized collection of network traffic logs, introducing:
- **Privacy Violations** (exposing proprietary banking/patient data)
- **Extreme Bandwidth Consumption** (streaming gigabytes of raw PCAPs)
- **Regulatory Non-Compliance** (violating GDPR, HIPAA, and financial regulations)

**FAI-IDS solves this through Federated Learning:**
- Each client organization trains a local PyTorch Deep Neural Network on its private network flows.
- Only encrypted/abstracted model parameter weights are transmitted to a central Flower Aggregation Server.
- The Flower Server performs **Federated Averaging (FedAvg)** to combine weights into an improved global model.
- The updated global model is synchronized back to every client for subsequent defense rounds.

```
+-----------------------------------------------------------------------------------+
|                            FAI-IDS ARCHITECTURE                                    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +------------------+     +------------------+     +--------------------------+   |
|  |  Bank Network    |     | Hospital Network |     | University / Tech Org    |   |
|  | (Local CICIDS)   |     | (Local CICIDS)   |     | (Local CICIDS)           |   |
|  | [PyTorch MLP]    |     | [PyTorch MLP]    |     | [PyTorch MLP]            |   |
|  +--------+---------+     +--------+---------+     +------------+-------------+   |
|           |                        |                            |                 |
|           | (Local Weights Only)   | (Local Weights Only)       |                 |
|           +-------------------+    |    +-----------------------+                 |
|                               v    v    v                                         |
|                    +--------------------------------+                             |
|                    |     Flower FedAvg Server       |                             |
|                    |  Aggregates Model Parameters   |                             |
|                    |   Evaluates Global Metrics     |                             |
|                    |   Saves Global Checkpoints     |                             |
|                    +---------------+----------------+                             |
|                                    |                                              |
|            +-----------------------+-----------------------+                      |
|            v                                               v                      |
|  +--------------------+                         +----------------------+          |
|  |   SQLite Database  |                         |  Streamlit Dashboard |          |
|  | - Clients          |                         | - Live Attack Counts |          |
|  | - TrainingLogs     |<========================| - Loss / Acc Curves  |          |
|  | - AttackLogs       |                         | - Confusion Matrix   |          |
|  | - GlobalModel      |                         | - Real-time Flow IDS |          |
|  | - Predictions      |                         | - Client Comparison  |          |
|  +--------------------+                         +----------------------+          |
+-----------------------------------------------------------------------------------+
```

---

## 2. Project Directory Structure

```
FAI_IDS/
│
├── server/
│   ├── server.py              # Flower Aggregation Server runner
│   └── strategy.py            # Custom FedAvg strategy with SQLite metric logging & checkpointing
│
├── clients/
│   ├── base_client.py         # Flower NumPyClient implementation with local PyTorch training
│   ├── client1.py             # Apex Bank Network node
│   ├── client2.py             # St. Jude Memorial Hospital node
│   └── client3.py             # Federal State University node
│
├── model/
│   ├── model.py               # PyTorch MLP (80-features -> 128 -> 64 -> 16 classes)
│   ├── train.py               # Local Adam optimizer training loop (CrossEntropyLoss)
│   └── evaluate.py            # Accuracy, Precision, Recall, F1, Confusion Matrix
│
├── data/
│   ├── generate_cicids_subset.py # Authentic 80-feature CICIDS2017 dataset generator
│   ├── preprocess.py          # Scaling, Label Encoding, and Non-IID Dirichlet partitioner
│   └── cicids_dataset.csv     # Extracted flow dataset
│
├── dashboard/
│   └── app.py                 # Modern Cybersecurity Command Center (Streamlit + Plotly)
│
├── database/
│   ├── schema.sql             # Relational schema for Clients, Logs, Models, and Predictions
│   └── db.py                  # Thread-safe SQLite data access layer
│
├── utils/
│   ├── helpers.py             # MITRE threat intelligence, mitigation playbooks, helpers
│   └── logger.py              # Standardized application logging
│
├── run_simulation.py          # End-to-end automated multi-client federated training orchestrator
├── requirements.txt           # Python dependency requirements
├── spec.md3.txt               # Single source of truth specification document
└── README.md                  # Complete documentation
```

---

## 3. Deep Learning Model Architecture (Spec Section 11)

| Layer | Specifications |
|---|---|
| **Input Layer** | 80 CICFlowMeter Network Traffic Features |
| **Hidden Layer 1** | 128 Neurons + ReLU Activation |
| **Dropout** | 0.3 Rate (Regularization against overfitting) |
| **Hidden Layer 2** | 64 Neurons + ReLU Activation |
| **Dropout** | 0.3 Rate |
| **Output Layer** | 15 Attack Classes + Benign (16 logits) |
| **Loss Function** | `CrossEntropyLoss` |
| **Optimizer** | `Adam` (learning rate: 0.001) |
| **Aggregation** | `FedAvg` (Federated Averaging) |

### Supported Attack Classes (CICIDS2017)
1. `BENIGN`
2. `DoS Hulk`
3. `PortScan`
4. `DDoS`
5. `DoS GoldenEye`
6. `FTP-Patator`
7. `SSH-Patator`
8. `DoS slowloris`
9. `DoS Slowhttptest`
10. `Botnet`
11. `Web Attack - Brute Force`
12. `Web Attack - XSS`
13. `Infiltration`
14. `Web Attack - Sql Injection`
15. `Heartbleed`

---

## 4. Quick Start Guide

### Step 1: Activate Virtual Environment & Install Dependencies
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Install required libraries
pip install -r requirements.txt
```

### Step 2: Run the Automated Federated Learning Simulation
Run 5 federated training rounds across Bank, Hospital, and University nodes:
```powershell
python run_simulation.py
```
This will:
- Initialize the SQLite database (`fai_ids.db`)
- Synthesize/preprocess the 80-feature CICIDS2017 dataset
- Distribute heterogeneous (non-IID) data partitions to the 3 organizations
- Execute 5 rounds of local PyTorch training and server FedAvg aggregation
- Save the global model checkpoint to `model/global_model.pt`
- Seed real-time intrusion events for the dashboard

### Step 3: Launch the Interactive Cyber Defense Dashboard
```powershell
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Distributed Network Deployment Mode (Client / Server)

If you wish to run the components across separate terminal sessions or networked devices:

1. **Terminal 1: Start Flower Aggregation Server**
   ```powershell
   python server/server.py --address 0.0.0.0:8080 --rounds 5 --min_clients 3
   ```
2. **Terminal 2: Launch Bank Client**
   ```powershell
   python clients/client1.py --server 127.0.0.1:8080
   ```
3. **Terminal 3: Launch Hospital Client**
   ```powershell
   python clients/client2.py --server 127.0.0.1:8080
   ```
4. **Terminal 4: Launch University Client**
   ```powershell
   python clients/client3.py --server 127.0.0.1:8080
   ```

---

## 6. Database Schema (SQLite)

- **`Clients`**: Registered organization profiles, IP address, status, dataset record counts.
- **`TrainingLogs`**: Per-round training loss, accuracy, and validation metrics for each organization.
- **`AttackLogs`**: Detailed intrusion records including attack type, severity, target port, protocol, and flow statistics.
- **`GlobalModel`**: Aggregated model version tracking, global accuracy, precision, recall, and F1 score.
- **`Predictions`**: Live flow testing audit trail with timestamps, input features, and confidence levels.
