"""
Local PyTorch Training Loop for FAI-IDS Clients
Spec Reference: Section 11 & 12 (Model Architecture & Local Training)
"""

import time
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import IntrusionDetectionNN


def train_local_epoch(model: IntrusionDetectionNN,
                      train_loader: DataLoader,
                      optimizer: torch.optim.Optimizer,
                      criterion: nn.Module,
                      device: torch.device) -> Tuple[float, float]:
    """
    Trains the model for one local epoch.
    Returns:
        (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def train_local_model(model: IntrusionDetectionNN,
                      train_loader: DataLoader,
                      val_loader: Optional[DataLoader] = None,
                      epochs: int = 3,
                      learning_rate: float = 0.001,
                      device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Executes multiple local epochs of training using Adam optimizer and CrossEntropyLoss.
    Returns training summary dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # Spec requirement: Loss: CrossEntropyLoss, Optimizer: Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    start_time = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    final_train_loss = 0.0
    final_train_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_local_epoch(model, train_loader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        final_train_loss = train_loss
        final_train_acc = train_acc

    val_loss, val_acc = 0.0, 0.0
    if val_loader:
        from model.evaluate import evaluate_model
        eval_metrics = evaluate_model(model, val_loader, device)
        val_loss = eval_metrics["loss"]
        val_acc = eval_metrics["accuracy"]
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    elapsed = time.time() - start_time

    return {
        "train_loss": float(final_train_loss),
        "train_accuracy": float(final_train_acc),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "duration_sec": float(elapsed),
        "history": history
    }
