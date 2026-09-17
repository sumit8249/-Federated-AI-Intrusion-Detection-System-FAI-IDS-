"""
Flower Federated Aggregation Server for FAI-IDS
Spec Reference: Section 4 (Flower Aggregation Server), Section 7 (Admin Functions), Section 17
"""

import sys
import os
import argparse
import flwr as fl

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db
from model.model import get_model
from server.strategy import FAIFedAvg, GLOBAL_MODEL_PATH
from flwr.common import ndarrays_to_parameters


def run_server(server_address: str = "0.0.0.0:8080", num_rounds: int = 5, min_clients: int = 3):
    """
    Initializes database and runs Flower FedAvg server.
    """
    init_db()
    print("=" * 60)
    print(" Federate AI Intrusion Detection System (FAI-IDS) Server")
    print("=" * 60)
    print(f"[*] Aggregation Strategy : FedAvg")
    print(f"[*] Training Rounds      : {num_rounds}")
    print(f"[*] Minimum Clients Req  : {min_clients}")
    print(f"[*] Server Listening on  : {server_address}")
    print("=" * 60)

    # Initial global model parameters
    initial_model = get_model()
    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())

    strategy = FAIFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"server_round": rnd, "local_epochs": 3},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    print("\n[Server] Federated aggregation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FAI-IDS Flower Server")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Address to bind server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--min_clients", type=int, default=3, help="Minimum clients required per round")
    args = parser.parse_args()

    run_server(server_address=args.address, num_rounds=args.rounds, min_clients=args.min_clients)
