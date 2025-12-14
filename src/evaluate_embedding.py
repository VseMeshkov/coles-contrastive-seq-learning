# evaluate_embeddings.py
"""
Downstream evaluation script for learned embeddings.

Trains a simple MLP classifier on top of frozen embeddings to evaluate
the quality of learned representations. Uses accuracy as the evaluation metric.
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
import yaml


class EmbeddingDataset(Dataset):
    """
    Dataset wrapper for embedding-target pairs.

    Args:
        embeddings: NumPy array of embeddings (N, D)
        targets: NumPy array of target labels (N,)
    """

    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_embeddings(path):
    """
    Load embeddings from parquet file and filter out missing targets.

    Args:
        path: Path to parquet file containing embeddings and targets

    Returns:
        tuple: (embeddings array, targets array)
    """
    df = pd.read_parquet(path)

    # Filter out samples with missing targets
    if "target" in df.columns:
        df = df[df["target"].notna()].reset_index(drop=True)

    # Extract embeddings and targets
    X = np.vstack(df["embedding"].values)
    y = df["target"].values.astype(np.int64)

    return X, y


def create_dataloader(X, y, batch_size, num_workers, shuffle=False):
    """
    Create DataLoader from embeddings and targets.

    Args:
        X: Embeddings array
        y: Targets array
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = EmbeddingDataset(torch.FloatTensor(X), torch.LongTensor(y))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_epoch(model, dataloader, optimizer, device):
    """
    Train classifier for one epoch.

    Args:
        model: Classifier model
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on
    """
    model.train()

    for embeddings, targets in dataloader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = model(embeddings, targets)["loss"]
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate classifier accuracy.

    Args:
        model: Classifier model
        dataloader: Evaluation data loader
        device: Device to evaluate on

    Returns:
        Accuracy score
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for embeddings, targets in dataloader:
        embeddings = embeddings.to(device)

        # Get predictions
        logits = model(embeddings, y=None)["logits"]
        predictions = torch.argmax(logits, dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())

    # Calculate accuracy
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    return accuracy_score(y_true, y_pred)


def save_checkpoint(model, filepath):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        filepath: Path to save checkpoint
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def save_results(results, filepath):
    """
    Save evaluation results to YAML file.

    Args:
        results: Dictionary of results
        filepath: Path to save results
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        yaml.dump(results, f)


@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(config: DictConfig):
    """
    Main evaluation function.

    Trains a linear classifier on embeddings and evaluates on test set.
    Uses train/val split with early stopping.

    Args:
        config: Hydra configuration object
    """
    set_seed(config.seed)
    output_checkpoint_dir = Path(
        f"{config.output.results_path}/{config.output.exp_name}/{config.output.run_name}"
    )
    # Load embeddings
    X_train_full, y_train_full = load_embeddings(config.data.train_embeddings_path)
    X_test, y_test = load_embeddings(config.data.test_embeddings_path)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.data.val_size,
        random_state=config.seed,
        stratify=y_train_full,
    )

    # Create dataloaders
    train_loader = create_dataloader(
        X_train,
        y_train,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        X_val,
        y_val,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=False,
    )
    test_loader = create_dataloader(
        X_test,
        y_test,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=False,
    )

    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = instantiate(config.model, input_size=input_size, num_classes=num_classes)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.optimizer.lr)

    # Training loop with early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(config.training.num_epochs):
        # Train and validate
        train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

            save_checkpoint(model, output_checkpoint_dir / "best_downstream_model.pth")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= config.training.early_stopping:
            break

    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(output_checkpoint_dir / "best_downstream_model.pth")
    )
    test_acc = evaluate(model, test_loader, device)

    # Save results
    results = {
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
    }
    output_path = (
        Path(config.output.results_path)
        / config.output.exp_name
        / config.output.run_name
        / "downstream_results.yaml"
    )
    save_results(results, output_path)

    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
