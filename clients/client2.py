"""
Federated Client 2: St. Jude Memorial Hospital
Spec Reference: Section 4 (Client Organizations - Hospital) & Section 17
"""

import sys
import os
import argparse
import flwr as fl

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import register_client
from data.preprocess import load_and_preprocess_dataset, partition_data_for_clients, create_dataloader
from clients.base_client import FAIClient


def start_client(server_address: str = "127.0.0.1:8080"):
    client_id = "client_002_hospital"
    org_name = "St. Jude Memorial Hospital"
    org_type = "Hospitals"

    print(f"[{org_name}] Initializing Hospital Intrusion Detection Node...")
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_dataset()
    partitions = partition_data_for_clients(X_train, y_train, num_clients=3, non_iid=True)

    # Client 2 uses partition 1
    X_local, y_local = partitions[1]
    train_loader = create_dataloader(X_local, y_local, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    register_client(
        client_id=client_id,
        organization_name=org_name,
        organization_type=org_type,
        dataset_records=len(X_local)
    )

    client = FAIClient(
        client_id=client_id,
        organization_name=org_name,
        train_loader=train_loader,
        val_loader=val_loader,
        local_epochs=3,
        lr=0.001
    )

    print(f"[{org_name}] Connecting to Flower Server at {server_address}...")
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Hospital Federated Client")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Flower server address")
    args = parser.parse_args()
    start_client(args.server)
