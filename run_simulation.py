"""
End-to-End Federated Simulation Orchestrator for FAI-IDS
Runs local federated training rounds without socket networking issues,
aggregates weights using FedAvg, logs all telemetry to SQLite,
and prepares the system for the interactive Streamlit dashboard.
"""

import sys
import os
import time
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import (
    init_db,
    register_client,
    log_training_round,
    log_global_model,
    log_attack,
    log_prediction
)
from model.model import get_model, ATTACK_CLASSES, NUM_FEATURES
from model.train import train_local_model
from model.evaluate import evaluate_model
from data.generate_cicids_subset import generate_realistic_cicids
from data.preprocess import load_and_preprocess_dataset, partition_data_for_clients, create_dataloader
from utils.helpers import get_threat_info


def run_federated_simulation(num_rounds: int = 5, local_epochs: int = 2, num_clients: int = 3):
    """
    Executes an end-to-end federated learning simulation:
    1. Generates CICIDS2017 dataset & partitions data across Bank, Hospital, and University.
    2. Initializes global model parameters.
    3. Runs FedAvg aggregation across federated rounds.
    4. Evaluates global model on global test set.
    5. Saves global model checkpoint.
    6. Injects initial realistic intrusion logs for dashboard verification.
    """
    print("=" * 70)
    print(" FEDERATED AI INTRUSION DETECTION SYSTEM (FAI-IDS) SIMULATION")
    print("=" * 70)

    # 1. Initialize DB
    init_db()
    print("[+] Database initialized at fai_ids.db")

    # 2. Check/Generate Dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cicids_dataset.csv")
    if not os.path.exists(dataset_path):
        print("[+] Generating CICIDS2017 realistic multi-class flow dataset...")
        generate_realistic_cicids(num_samples=10000, output_path=dataset_path)

    # 3. Preprocess & Partition Data
    print("[+] Preprocessing dataset and standardizing 80 flow features...")
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_dataset(csv_path=dataset_path)
    client_partitions = partition_data_for_clients(X_train, y_train, num_clients=num_clients, non_iid=True)

    org_profiles = [
        {"id": "client_001_bank", "name": "Apex Bank Network", "type": "Banking", "ip": "10.0.1.15"},
        {"id": "client_002_hospital", "name": "St. Jude Memorial Hospital", "type": "Hospitals", "ip": "10.0.2.42"},
        {"id": "client_003_university", "name": "Federal State University", "type": "Universities", "ip": "10.0.3.88"},
    ]

    client_loaders = []
    for idx, org in enumerate(org_profiles):
        X_local, y_local = client_partitions[idx]
        register_client(
            client_id=org["id"],
            organization_name=org["name"],
            organization_type=org["type"],
            ip_address=org["ip"],
            dataset_records=len(X_local)
        )
        loader = create_dataloader(X_local, y_local, batch_size=64, shuffle=True)
        client_loaders.append(loader)
        print(f"    - Registered Node: {org['name']} ({org['type']}) with {len(X_local)} private flows")

    test_loader = create_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    # 4. Initialize Global PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Global Model Initialized on device: {device}")
    global_model = get_model().to(device)
    global_weights = global_model.get_weights()

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    global_model_path = os.path.join(model_dir, "global_model.pt")

    # 5. Federated Training Loop (FedAvg)
    print("\n" + "-" * 70)
    print(f" Starting Federated Training: {num_rounds} Rounds, {num_clients} Organizations")
    print("-" * 70)

    for rnd in range(1, num_rounds + 1):
        round_start = time.time()
        print(f"\n>>> [Round {rnd}/{num_rounds}] Distributing global model to clients...")

        client_weights_list = []
        client_sample_counts = []

        for idx, org in enumerate(org_profiles):
            # Local client trains on local data starting from global weights
            local_model = get_model().to(device)
            local_model.set_weights(global_weights)

            summary = train_local_model(
                model=local_model,
                train_loader=client_loaders[idx],
                val_loader=test_loader,
                epochs=local_epochs,
                learning_rate=0.001,
                device=device
            )

            n_samples = len(client_partitions[idx][0])
            client_weights_list.append(local_model.get_weights())
            client_sample_counts.append(n_samples)

            # Record round telemetry in SQLite
            log_training_round(
                round_number=rnd,
                client_id=org["id"],
                train_loss=summary["train_loss"],
                train_accuracy=summary["train_accuracy"],
                val_loss=summary["val_loss"],
                val_accuracy=summary["val_accuracy"],
                samples_count=n_samples,
                duration_sec=summary["duration_sec"]
            )

            print(f"    * {org['name']}: Train Acc={summary['train_accuracy']*100:.2f}%, Loss={summary['train_loss']:.4f}")

        # Server Aggregation via FedAvg
        print("    * [Server] Aggregating local client weights using FedAvg...")
        total_samples = sum(client_sample_counts)
        new_global_weights = []

        for layer_idx in range(len(global_weights)):
            layer_avg = sum(
                client_weights_list[c][layer_idx] * (client_sample_counts[c] / total_samples)
                for c in range(num_clients)
            )
            new_global_weights.append(layer_avg)

        global_weights = new_global_weights
        global_model.set_weights(global_weights)

        # Evaluate aggregated model
        eval_res = evaluate_model(global_model, test_loader, device=device)
        elapsed = time.time() - round_start

        # Save checkpoint
        torch.save(global_model.state_dict(), os.path.join(model_dir, f"global_model_round_{rnd}.pt"))
        torch.save(global_model.state_dict(), global_model_path)

        # Log Global Model metrics
        log_global_model(
            round_number=rnd,
            aggregated_accuracy=eval_res["accuracy"],
            aggregated_loss=eval_res["loss"],
            precision_score=eval_res["precision"],
            recall_score=eval_res["recall"],
            f1_score=eval_res["f1_score"],
            participating_clients=num_clients,
            weights_path=global_model_path
        )

        print(f"    => Round {rnd} Complete ({elapsed:.2f}s) | Global Acc: {eval_res['accuracy']*100:.2f}% | Loss: {eval_res['loss']:.4f} | F1: {eval_res['f1_score']:.4f}")

    # 6. Seed Sample Intrusion Logs for Realistic Dashboard Experience
    print("\n[+] Generating initial intrusion telemetry logs for dashboard...")
    sample_threats = [
        {"client": "client_001_bank", "org": "Apex Bank Network", "attack": "DDoS", "src": "194.26.29.11", "port": 443, "proto": "TCP", "conf": 0.98},
        {"client": "client_001_bank", "org": "Apex Bank Network", "attack": "SSH-Patator", "src": "45.142.212.8", "port": 22, "proto": "TCP", "conf": 0.94},
        {"client": "client_002_hospital", "org": "St. Jude Memorial Hospital", "attack": "DoS Hulk", "src": "185.220.101.4", "port": 80, "proto": "TCP", "conf": 0.96},
        {"client": "client_002_hospital", "org": "St. Jude Memorial Hospital", "attack": "PortScan", "src": "103.152.18.2", "port": 445, "proto": "TCP", "conf": 0.91},
        {"client": "client_003_university", "org": "Federal State University", "attack": "Web Attack - Sql Injection", "src": "89.248.165.7", "port": 8080, "proto": "TCP", "conf": 0.97},
        {"client": "client_003_university", "org": "Federal State University", "attack": "Botnet", "src": "178.62.204.19", "port": 6667, "proto": "TCP", "conf": 0.93},
    ]

    for threat in sample_threats:
        tinfo = get_threat_info(threat["attack"])
        log_attack(
            client_id=threat["client"],
            organization_name=threat["org"],
            attack_type=threat["attack"],
            severity=tinfo["severity"],
            confidence=threat["conf"],
            source_ip=threat["src"],
            destination_port=threat["port"],
            protocol=threat["proto"],
            flow_duration=12500.0,
            total_packets=350,
            total_bytes=142000.0,
            flow_summary=f"Automated intrusion detected matching {threat['attack']} pattern",
            status="Alerted"
        )

    print("\n" + "=" * 70)
    print(" FEDERATED SIMULATION COMPLETED SUCCESSFULLY!")
    print(f" Final Global Model Accuracy: {eval_res['accuracy']*100:.2f}%")
    print(f" Final Global Model F1-Score: {eval_res['f1_score']:.4f}")
    print(f" Model checkpoint saved to: {global_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_federated_simulation(num_rounds=5, local_epochs=2, num_clients=3)
