# inference.py
"""
Inference script for extracting embeddings from trained CoLES model.

Loads a trained model checkpoint and extracts embeddings for downstream tasks.
Saves embeddings along with targets and client IDs (if available) to parquet format.
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datasets import prepare_age_pred_data
from train import set_seed


def load_model(config, device):
    """
    Load trained model from checkpoint.

    Handles different checkpoint formats:
    - Full checkpoint with model_state_dict key
    - Direct state dict for seq_encoder
    - Direct state dict for full model

    Args:
        config: Configuration containing model architecture and checkpoint path
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Initialize model architecture
    model = instantiate(config.model)

    # Load checkpoint
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        # Checkpoint contains full training state
        model.seq_encoder.load_state_dict(checkpoint["model_state_dict"])
    elif hasattr(model, "seq_encoder"):
        # Direct state dict for seq_encoder
        model.seq_encoder.load_state_dict(checkpoint)
    else:
        # Direct state dict for full model
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from sequences using trained model.

    Processes all batches in dataloader and extracts:
    - Embeddings from model
    - Targets (if available in batch)
    - Client IDs (if available in batch)

    Args:
        model: Trained CoLES model
        dataloader: DataLoader providing sequences
        device: Device to run inference on

    Returns:
        DataFrame with columns: embedding, target (optional), client_id (optional)
    """
    model.eval()

    all_embeddings = []
    all_targets = []
    all_client_ids = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        batch = batch.to(device)

        # Extract embeddings
        output = model(batch)

        if not isinstance(output, torch.Tensor):
            raise ValueError("Model output must be a torch.Tensor")

        all_embeddings.append(output.cpu().numpy())

        # Extract targets if available
        if "target" in batch.payload:
            targets = batch.payload["target"]
            # Handle both scalar and sequence targets
            if len(targets.shape) > 1:
                targets = targets[:, 0]
            all_targets.append(targets.cpu().numpy())

        # Extract client IDs if available
        if "client_id" in batch.payload:
            client_ids = batch.payload["client_id"]
            # Handle both scalar and sequence IDs
            if len(client_ids.shape) > 1:
                client_ids = client_ids[:, 0]
            all_client_ids.append(client_ids.cpu().numpy())

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    num_samples = len(embeddings)

    # Create DataFrame with embeddings as list of arrays
    df = pd.DataFrame({"embedding": [embeddings[i] for i in range(num_samples)]})

    # Add targets if available (using nullable integer type)
    if all_targets:
        targets = np.concatenate(all_targets, axis=0)
        df["target"] = pd.Series(targets, dtype="Int64")

    # Add client IDs if available (using nullable integer type)
    if all_client_ids:
        client_ids = np.concatenate(all_client_ids, axis=0)
        df["client_id"] = pd.Series(client_ids, dtype="Int64")

    return df


def save_embeddings(df, output_path):
    """
    Save embeddings DataFrame to parquet file.

    Args:
        df: DataFrame containing embeddings
        output_path: Path to save parquet file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(config: DictConfig):
    """
    Main inference function.

    Pipeline:
    1. Load test data
    2. Load trained model from checkpoint
    3. Extract embeddings
    4. Save to parquet file

    Args:
        config: Hydra configuration object
    """
    set_seed(config.seed)

    # Load test data (no train/val needed for inference)
    test_dataloader = prepare_age_pred_data(config, need_train_val=False)

    # Load trained model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    # Extract embeddings
    embeddings_df = extract_embeddings(model, test_dataloader, device)

    # Save results
    output_path = (
        Path(config.output.results_path)
        / config.output.exp_name
        / config.output.run_name
        / "inference_results"
        / f"{config.output.embeddings_filename}.parquet"
    )
    save_embeddings(embeddings_df, output_path)

    print(f"Extracted embeddings: {embeddings_df.shape[0]} samples")
    print(f"Embedding dimension: {len(embeddings_df['embedding'].iloc[0])}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
