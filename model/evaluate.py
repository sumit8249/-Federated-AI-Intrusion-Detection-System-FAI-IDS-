"""
Evaluation Metrics Module for FAI-IDS
Calculates Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
Spec Reference: Section 15 (Evaluation Metrics)
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from model.model import IntrusionDetectionNN, ATTACK_CLASSES


def evaluate_model(model: IntrusionDetectionNN,
                   data_loader: DataLoader,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Evaluates PyTorch model on a dataset loader and computes all standard intrusion metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    all_probs = []
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    loss = running_loss / max(total_samples, 1)
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) > 0 else 0.0
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) > 0 else 0.0

    # Multi-class confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ATTACK_CLASSES))))

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "total_samples": int(total_samples),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "targets": y_true.tolist()
    }
