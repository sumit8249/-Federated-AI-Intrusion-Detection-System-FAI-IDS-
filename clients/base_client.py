"""
Base Federated Client Implementation using Flower Framework
Spec Reference: Section 4, 7, 11 (Client Functions & Architecture)
"""

import time
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import flwr as fl
import torch
from torch.utils.data import DataLoader

from model.model import IntrusionDetectionNN, get_model
from model.train import train_local_model
from model.evaluate import evaluate_model
from database.db import log_training_round, update_client_status


class FAIClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing a private organization node (Bank, Hospital, University, etc.)
    Participates in federated learning rounds without sharing raw packet data.
    """
    def __init__(self, client_id: str, organization_name: str,
                 train_loader: DataLoader, val_loader: DataLoader,
                 local_epochs: int = 3, lr: float = 0.001):
        self.client_id = client_id
        self.organization_name = organization_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model().to(self.device)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns local model weights as NumPy arrays."""
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Receives aggregated global model parameters from server,
        performs local PyTorch training on private organization flows,
        logs local metrics to database, and returns updated parameters.
        """
        # Load server weights into local model
        self.model.set_weights(parameters)
        update_client_status(self.client_id, "Training")

        current_round = int(config.get("server_round", 1))
        epochs = int(config.get("local_epochs", self.local_epochs))

        # Local training
        summary = train_local_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=self.lr,
            device=self.device
        )

        num_samples = len(self.train_loader.dataset)

        # Log round to SQLite database
        log_training_round(
            round_number=current_round,
            client_id=self.client_id,
            train_loss=summary["train_loss"],
            train_accuracy=summary["train_accuracy"],
            val_loss=summary["val_loss"],
            val_accuracy=summary["val_accuracy"],
            samples_count=num_samples,
            duration_sec=summary["duration_sec"]
        )

        update_client_status(self.client_id, "Active")

        metrics = {
            "train_loss": summary["train_loss"],
            "train_accuracy": summary["train_accuracy"],
            "val_loss": summary["val_loss"],
            "val_accuracy": summary["val_accuracy"]
        }

        return self.model.get_weights(), num_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates the aggregated global model on local client validation partition.
        """
        self.model.set_weights(parameters)
        eval_metrics = evaluate_model(self.model, self.val_loader, device=self.device)

        num_samples = len(self.val_loader.dataset)
        return (
            eval_metrics["loss"],
            num_samples,
            {
                "accuracy": eval_metrics["accuracy"],
                "f1_score": eval_metrics["f1_score"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"]
            }
        )
