"""
PyTorch Deep Learning Model Architecture for FAI-IDS
Spec Reference: Section 11 (Machine Learning Model)

Architecture:
- Input Layer: 80 Network Features
- Hidden Layer 1: 128 Neurons + ReLU
- Hidden Layer 2: 64 Neurons + ReLU
- Dropout: 0.3
- Output Layer: Multi-class Attack Prediction (15 Attack Classes + Benign = 16)
- Loss: CrossEntropyLoss
- Optimizer: Adam
"""

import collections
from typing import List, OrderedDict
import numpy as np
import torch
import torch.nn as nn

# The 15 attack classes + BENIGN defined in CICIDS2017 specification
ATTACK_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Botnet",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Heartbleed"
]

NUM_FEATURES = 80
NUM_CLASSES = len(ATTACK_CLASSES)


class IntrusionDetectionNN(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Intrusion Detection.
    Conforms strictly to Section 11 of spec.md3.txt.
    """
    def __init__(self, input_dim: int = NUM_FEATURES, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
        super(IntrusionDetectionNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Hidden Layer 1: 80 -> 128 Neurons + ReLU
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Hidden Layer 2: 128 -> 64 Neurons + ReLU
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Output Layer: 64 -> num_classes logits
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.output_layer(x)
        return logits

    def get_weights(self) -> List[np.ndarray]:
        """
        Extracts PyTorch state dict weights into a list of NumPy arrays for Flower.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Loads aggregated NumPy parameter weights from Flower into the PyTorch model.
        """
        params_dict = zip(self.state_dict().keys(), weights)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)


def get_model(input_dim: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> IntrusionDetectionNN:
    """Factory method to instantiate a new IntrusionDetectionNN."""
    return IntrusionDetectionNN(input_dim=input_dim, num_classes=num_classes)
