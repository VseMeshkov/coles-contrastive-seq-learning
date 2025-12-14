# datasets.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.model_selection import train_test_split

from ptls.data_load.datasets import MemoryMapDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames.coles import ColesDataset
from ptls.data_load.padded_batch import PaddedBatch


class UniversalCollator:
    """
    Universal collate function for standard data (without splits).

    Handles both sequence and scalar features, automatically infers dtypes,
    and creates properly padded batches for PyTorch models.

    Args:
        categorical_features: List of categorical feature names (will be cast to torch.long)
        numeric_features: List of numeric feature names (will be cast to torch.float32)
    """

    def __init__(self, categorical_features=None, numeric_features=None):
        self.categorical_features = set(categorical_features or [])
        self.numeric_features = set(numeric_features or [])

    def _infer_dtype(self, key, value):
        """
        Infer appropriate dtype and padding value for a feature.

        Args:
            key: Feature name
            value: Sample value from the feature

        Returns:
            tuple: (torch dtype, padding value)
        """
        if key in self.categorical_features:
            return torch.long, 0
        if key in self.numeric_features:
            return torch.float32, 0.0

        # Auto-detect from first valid value
        if value is None or (isinstance(value, (list, np.ndarray)) and len(value) == 0):
            return torch.float32, 0.0

        sample = value[0] if isinstance(value, (list, np.ndarray)) else value
        if isinstance(sample, (int, np.integer)):
            return torch.long, 0
        return torch.float32, 0.0

    def __call__(self, batch):
        """
        Collate a batch of samples into a PaddedBatch.

        Args:
            batch: List of dicts, each containing feature arrays

        Returns:
            PaddedBatch with padded sequences and proper lengths
        """
        batch_size = len(batch)
        keys = batch[0].keys()

        # Collect all values by key
        payload = {key: [sample[key] for sample in batch] for key in keys}

        seq_lens = []
        padded_payload = {}

        for key, arrays in payload.items():
            # Find first valid value
            first_valid = next((x for x in arrays if x is not None), None)
            if first_valid is None:
                continue  # Skip fields with all None values

            # Determine if sequence or scalar
            is_sequence = isinstance(first_valid, (list, np.ndarray))

            if is_sequence:
                # Process sequences
                if not seq_lens:
                    seq_lens = [
                        (
                            len(x)
                            if x is not None and isinstance(x, (list, np.ndarray))
                            else 0
                        )
                        for x in arrays
                    ]

                max_len = max(
                    (
                        len(x)
                        for x in arrays
                        if x is not None and isinstance(x, (list, np.ndarray))
                    ),
                    default=0,
                )
                if max_len == 0:
                    continue

                dtype, pad_value = self._infer_dtype(key, first_valid)
                padded_tensor = torch.full(
                    (batch_size, max_len), pad_value, dtype=dtype
                )

                for i, arr in enumerate(arrays):
                    if arr is not None and len(arr) > 0:
                        padded_tensor[i, : len(arr)] = torch.tensor(arr, dtype=dtype)

                padded_payload[key] = padded_tensor
            else:
                # Process scalars
                dtype, default_value = self._infer_dtype(key, first_valid)
                values = [x if x is not None else default_value for x in arrays]
                padded_payload[key] = torch.tensor(values, dtype=dtype)

        seq_lens = seq_lens or [1] * batch_size
        return PaddedBatch(
            payload=padded_payload, length=torch.tensor(seq_lens, dtype=torch.long)
        )


class CoLESCollator:
    """
    Collate function for CoLES (Contrastive Learning for Event Sequences).

    Processes batches where each sample contains multiple splits of sequences,
    creating proper targets for contrastive learning.

    Args:
        split_count: Number of splits per sample
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
    """

    def __init__(self, split_count=5, categorical_features=None, numeric_features=None):
        self.split_count = split_count
        self.categorical_features = set(categorical_features or [])
        self.numeric_features = set(numeric_features or [])

    def _infer_dtype(self, key, value):
        """
        Infer appropriate dtype and padding value for a feature.

        Args:
            key: Feature name
            value: Sample value from the feature

        Returns:
            tuple: (torch dtype, padding value)
        """
        if key in self.categorical_features:
            return torch.long, 0
        if key in self.numeric_features:
            return torch.float32, 0.0

        sample = value[0] if isinstance(value, (list, np.ndarray)) else value
        if isinstance(sample, (int, np.integer)):
            return torch.long, 0
        return torch.float32, 0.0

    def __call__(self, batch):
        """
        Collate a batch of samples with splits into a PaddedBatch with contrastive targets.

        Args:
            batch: List of samples, each containing split_count dicts

        Returns:
            tuple: (PaddedBatch, targets) where targets are indices for contrastive loss
        """
        batch_size = len(batch)

        # Flatten all splits into a single list
        all_splits = [split for sample in batch for split in sample]
        keys = all_splits[0].keys()

        # Collect all values
        payload = {key: [split[key] for split in all_splits] for key in keys}

        padded_payload = {}
        seq_lens = []

        for key, arrays in payload.items():
            is_sequence = isinstance(arrays[0], (list, np.ndarray))

            if is_sequence:
                if not seq_lens:
                    seq_lens = [len(arr) for arr in arrays]

                max_len = max(len(arr) for arr in arrays)
                dtype, pad_value = self._infer_dtype(key, arrays[0])
                padded_tensor = torch.full(
                    (len(arrays), max_len), pad_value, dtype=dtype
                )

                for i, arr in enumerate(arrays):
                    padded_tensor[i, : len(arr)] = torch.tensor(arr, dtype=dtype)

                padded_payload[key] = padded_tensor
            else:
                dtype, _ = self._infer_dtype(key, arrays[0])
                padded_payload[key] = torch.tensor(arrays, dtype=dtype)

        padded_batch = PaddedBatch(
            payload=padded_payload, length=torch.tensor(seq_lens, dtype=torch.long)
        )
        padded_batch.split_count = self.split_count
        padded_batch.batch_size = batch_size

        # Create contrastive targets
        target = torch.arange(batch_size, dtype=torch.long).repeat_interleave(
            self.split_count
        )
        return padded_batch, target


class UniversalDataset(Dataset):
    """
    Simple wrapper dataset for list of dictionaries.

    Args:
        data: List of dicts containing transaction sequences
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_coles_dataloader(
    coles_dataset,
    batch_size,
    num_workers,
    split_count,
    categorical_features=None,
    numeric_features=None,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
):
    """
    Create DataLoader for CoLES training.

    Args:
        coles_dataset: ColesDataset instance with splits
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        split_count: Number of splits per sample
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        drop_last: Whether to drop last incomplete batch
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader configured for CoLES training
    """
    collator = CoLESCollator(
        split_count=split_count,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    return DataLoader(
        coles_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_inference_dataloader(
    data,
    batch_size,
    num_workers,
    categorical_features=None,
    numeric_features=None,
    shuffle=False,
    pin_memory=True,
):
    """
    Create DataLoader for inference (no splits).

    Args:
        data: List of dicts containing transaction sequences
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        shuffle: Whether to shuffle data (typically False for inference)
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader configured for inference
    """
    dataset = UniversalDataset(data)
    collator = UniversalCollator(
        categorical_features=categorical_features, numeric_features=numeric_features
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )


def _extract_features(data, squeeze_col="feature_arrays", drop_cols=None):
    """
    Extract features from nested dict structure.

    Args:
        data: Dict with nested feature_arrays
        squeeze_col: Name of column containing nested features
        drop_cols: List of columns to drop

    Returns:
        Dict with flattened features
    """
    drop_cols = drop_cols or []

    for key in data[squeeze_col].keys():
        data[key] = data[squeeze_col][key]

    for col in drop_cols:
        if col in data:
            data.pop(col)

    data.pop(squeeze_col)
    return data


def prepare_age_pred_data(config, need_train_val=True):
    """
    Prepare age prediction data with train/val/test splits.

    Loads transaction data from pickle files, applies feature extraction,
    creates train/validation split, and wraps everything in DataLoaders.

    Args:
        config: Hydra config containing:
            - data.categorical_features: List of categorical features
            - data.numeric_features: List of numeric features
            - train_data.txn_pickle_path: Path to training data
            - test_data.txn_pickle_path: Path to test data
            - train_data/val_data configs for splits and batching
        need_train_val: If True, returns (train, val, test) loaders,
                       if False, returns only test loader for inference

    Returns:
        If need_train_val=True: (train_dataloader, val_dataloader, test_dataloader)
        If need_train_val=False: test_dataloader
    """
    categorical_features = config.data.get("categorical_features", None)
    numeric_features = config.data.get("numeric_features", None)

    if need_train_val:
        # Load and process training data
        with open(config.train_data.txn_pickle_path, "rb") as f:
            train_data = pickle.load(f)

        train_data = [_extract_features(d) for d in train_data]

        # Split train/validation
        train_data, val_data = train_test_split(
            train_data,
            test_size=config.val_data.get("val_split", 0.1),
            random_state=config.seed,
        )

        # Create train dataloader with CoLES splits
        splitter = SampleSlices(
            split_count=config.train_data.split_count,
            cnt_min=config.train_data.cnt_min,
            cnt_max=config.train_data.cnt_max,
        )
        train_coles_dataset = ColesDataset(
            MemoryMapDataset(train_data), splitter=splitter
        )

        train_dataloader = create_coles_dataloader(
            train_coles_dataset,
            config.train_data.batch_size,
            config.train_data.num_workers,
            config.train_data.split_count,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        # Create validation dataloader with CoLES splits
        splitter = SampleSlices(
            split_count=config.val_data.split_count,
            cnt_min=config.val_data.cnt_min,
            cnt_max=config.val_data.cnt_max,
        )
        val_coles_dataset = ColesDataset(MemoryMapDataset(val_data), splitter=splitter)

        val_dataloader = create_coles_dataloader(
            val_coles_dataset,
            config.val_data.batch_size,
            config.val_data.num_workers,
            config.val_data.split_count,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    # Load and process test data (always loaded)
    with open(config.test_data.txn_pickle_path, "rb") as f:
        test_data = pickle.load(f)

    test_data = [_extract_features(d) for d in test_data]

    # Create test dataloader (inference mode - no splits)
    test_dataloader = create_inference_dataloader(
        test_data,
        config.test_data.batch_size,
        config.test_data.num_workers,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        shuffle=False,
        pin_memory=True,
    )

    if need_train_val:
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return test_dataloader
