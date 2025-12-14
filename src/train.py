# train.py
"""
Training script for CoLES (Contrastive Learning for Event Sequences) model.

This script handles the complete training pipeline including:
- Data loading and preparation
- Model training with gradient clipping
- Validation with early stopping
- Model checkpointing
- Metrics tracking
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from datasets import prepare_age_pred_data


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and CUDA.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """
    Train model for one epoch.

    Args:
        model: CoLES model instance
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on (cpu/cuda)
        grad_clip: Maximum gradient norm for clipping

    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch, targets in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass through CoLES model
        embeddings, labels = model.shared_step(batch, targets)
        loss = model._loss(embeddings, labels)

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.inference_mode()
def validate(model, dataloader, device):
    """
    Validate model on validation set.

    Args:
        model: CoLES model instance
        dataloader: Validation data loader
        device: Device to validate on (cpu/cuda)

    Returns:
        tuple: (average_loss, average_metric)
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    for batch, targets in tqdm(dataloader, desc="Validation"):
        batch = batch.to(device)
        targets = targets.to(device)

        # Forward pass
        embeddings, labels = model.shared_step(batch, targets)
        loss = model._loss(embeddings, labels)

        # Calculate validation metric
        metric = model._validation_metric(embeddings, labels)
        metric_value = metric.item() if torch.is_tensor(metric) else metric

        total_loss += loss.item()
        total_metric += metric_value
        num_batches += 1

    return total_loss / num_batches, total_metric / num_batches


def save_checkpoint(model, optimizer, epoch, val_metric, config, filepath):
    """
    Save model checkpoint with metadata.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        val_metric: Validation metric value
        config: Training configuration
        filepath: Path to save checkpoint
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.seq_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metric": val_metric,
            "config": OmegaConf.to_container(config, resolve=True),
        },
        filepath,
    )


def init_extra_pathes(config):
    exp_name = config.output.exp_name
    run_name = config.output.run_name

    results_path = Path(config.output.results_path)

    checkpoint_dir = f"{results_path}/{exp_name}/{run_name}/checkpoints"
    train_logs_path = f"{results_path}/{exp_name}/{run_name}/train_logs.yaml"

    return checkpoint_dir, train_logs_path


def train(config):
    """
    Main training function.

    Handles complete training loop including:
    - Data loading
    - Model initialization
    - Training/validation iterations
    - Early stopping
    - Checkpoint saving
    - Metrics tracking

    Args:
        config: Hydra configuration object
    """
    device = torch.device(config.device)
    checkpoint_dir, train_logs_path = init_extra_pathes(config)

    train_loader, val_loader, _ = prepare_age_pred_data(config, need_train_val=True)

    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.training.optimizer)(model.parameters())

    best_val_metric = -float("inf")
    epochs_no_improve = 0
    metrics_history = {"train_loss": [], "valid_loss": [], "valid_metric": []}

    for epoch in range(config.training.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, grad_clip=config.training.grad_clip
        )

        val_loss, val_metric = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid Metric: {val_metric:.4f}")

        metrics_history["train_loss"].append(float(train_loss))
        metrics_history["valid_loss"].append(float(val_loss))
        metrics_history["valid_metric"].append(float(val_metric))

        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0

            checkpoint_path = f"{checkpoint_dir}/{epoch}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_metric, config, checkpoint_path
            )
            print(f"Best model saved! (metric: {best_val_metric:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.training.early_stopping:
            print(f"Early stopping after {epoch+1} epochs")
            break

    with open(train_logs_path, "w") as f:
        yaml.safe_dump(metrics_history, f)

    print(f"\nTraining completed. Best validation metric: {best_val_metric:.4f}")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig):
    set_seed(config.seed)
    train(config)


if __name__ == "__main__":
    main()
