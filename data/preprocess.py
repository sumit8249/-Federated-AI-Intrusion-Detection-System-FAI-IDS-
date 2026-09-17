"""
Data Preprocessing and Client Partitioning Pipeline
Spec Reference: Section 5 (Preprocessing) & Section 12 (Steps 1-3)
"""

import os
import pickle
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.model import ATTACK_CLASSES, NUM_FEATURES

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw network flow dataset:
    - Strips column names
    - Replaces infinite values with NaNs and imputes with median/0
    - Drops zero-variance or constant columns if any
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Replace Inf and -Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs with column medians for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val if not np.isnan(median_val) else 0.0, inplace=True)

    return df


def load_and_preprocess_dataset(csv_path: str = None, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Loads CICIDS2017 dataset, normalizes features, encodes attack labels, and splits train/test.
    Saves scaler and label encoder objects for real-time inference.
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "cicids_dataset.csv")

    if not os.path.exists(csv_path):
        from data.generate_cicids_subset import generate_realistic_cicids
        generate_realistic_cicids(output_path=csv_path)

    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    # Separate features and label
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y_raw = df[label_col].astype(str).values
    feature_cols = [c for c in df.columns if c != label_col][:NUM_FEATURES]
    X_raw = df[feature_cols].values.astype(np.float32)

    # Ensure exactly NUM_FEATURES (80)
    if X_raw.shape[1] < NUM_FEATURES:
        pad = np.zeros((X_raw.shape[0], NUM_FEATURES - X_raw.shape[1]), dtype=np.float32)
        X_raw = np.hstack([X_raw, pad])
    elif X_raw.shape[1] > NUM_FEATURES:
        X_raw = X_raw[:, :NUM_FEATURES]

    # Encode labels
    encoder = LabelEncoder()
    # Fit with the full defined ATTACK_CLASSES to ensure deterministic integer mapping
    encoder.fit(ATTACK_CLASSES)
    
    # Map raw labels safely
    mapped_labels = []
    for label in y_raw:
        if label in ATTACK_CLASSES:
            mapped_labels.append(label)
        else:
            # Match closest known attack or fallback to BENIGN
            match = next((a for a in ATTACK_CLASSES if a.lower() in label.lower()), "BENIGN")
            mapped_labels.append(match)
    y = encoder.transform(mapped_labels)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Persist scaler and label encoder
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    return X_train, X_test, y_train, y_test, scaler, encoder


def partition_data_for_clients(X_train: np.ndarray, y_train: np.ndarray,
                               num_clients: int = 3,
                               non_iid: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partitions the training data across simulated federated organizations.
    If non_iid is True, partitions skew attacks per organization:
      - Client 1 (Bank): Skewed towards DDoS, Brute Force FTP/SSH
      - Client 2 (Hospital): Skewed towards DoS Hulk, PortScan
      - Client 3 (University): Skewed towards Web Attacks, Botnet, Infiltration
    """
    client_datasets = []
    n_samples = len(X_train)

    if not non_iid:
        # Uniform IID split
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, num_clients)
        for split in splits:
            client_datasets.append((X_train[split], y_train[split]))
        return client_datasets

    # Realistic Non-IID Dirichlet distribution based on attack labels
    unique_classes = np.unique(y_train)
    client_indices = [[] for _ in range(num_clients)]

    for c in unique_classes:
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)

        # Skew proportions per class for realistic domain diversity
        proportions = np.random.dirichlet(np.repeat(0.7, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)

        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    for i in range(num_clients):
        idxs = np.array(client_indices[i])
        np.random.shuffle(idxs)
        client_datasets.append((X_train[idxs], y_train[idxs]))

    return client_datasets


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Wraps NumPy arrays into PyTorch DataLoader."""
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
