"""
Custom Federated Averaging Strategy with Checkpointing and Database Logging
Spec Reference: Section 4, 11 (Federated Algorithm - FedAvg) & Section 14 (GlobalModel)
"""

import os
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from model.model import get_model
from database.db import log_global_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pt")


class FAIFedAvg(FedAvg):
    """
    Enhanced FedAvg strategy that:
    1. Computes weighted averages of client models.
    2. Logs global accuracy, loss, precision, recall, and F1 to SQLite.
    3. Saves PyTorch checkpoints of the aggregated global model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_accuracy = 0.0
        os.makedirs(MODEL_DIR, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates model weights using FedAvg and saves the global PyTorch model."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if parameters_aggregated is not None:
            # Convert Flower Parameters to NumPy ndarrays
            ndarrays = parameters_to_ndarrays(parameters_aggregated)

            # Load into PyTorch model and save checkpoint
            global_model = get_model()
            global_model.set_weights(ndarrays)

            checkpoint_path = os.path.join(MODEL_DIR, f"global_model_round_{server_round}.pt")
            torch.save(global_model.state_dict(), checkpoint_path)
            # Also save latest
            torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"[Server] Round {server_round}: Aggregated weights saved to {checkpoint_path}")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregates evaluation metrics from all clients and logs to SQLite GlobalModel table."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return loss_aggregated, metrics_aggregated

        # Weighted calculation of metrics across client test sets
        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / max(total_examples, 1)

        weighted_acc = sum(res.metrics.get("accuracy", 0.0) * res.num_examples for _, res in results) / max(total_examples, 1)
        weighted_f1 = sum(res.metrics.get("f1_score", 0.0) * res.num_examples for _, res in results) / max(total_examples, 1)
        weighted_prec = sum(res.metrics.get("precision", 0.0) * res.num_examples for _, res in results) / max(total_examples, 1)
        weighted_rec = sum(res.metrics.get("recall", 0.0) * res.num_examples for _, res in results) / max(total_examples, 1)

        # Log to SQLite Database
        log_global_model(
            round_number=server_round,
            aggregated_accuracy=weighted_acc,
            aggregated_loss=weighted_loss,
            precision_score=weighted_prec,
            recall_score=weighted_rec,
            f1_score=weighted_f1,
            participating_clients=len(results),
            weights_path=GLOBAL_MODEL_PATH
        )

        print(f"[Server] Round {server_round} Evaluation: Global Acc={weighted_acc*100:.2f}%, Loss={weighted_loss:.4f}, F1={weighted_f1:.4f}")

        custom_metrics = {
            "accuracy": weighted_acc,
            "f1_score": weighted_f1,
            "precision": weighted_prec,
            "recall": weighted_rec
        }

        return weighted_loss, custom_metrics
