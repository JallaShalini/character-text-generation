"""
PyTorch Dataset and DataLoader utilities for character-level text generation.
Handles sequence creation, batching, and train/test splitting.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List


class CharacterDataset(Dataset):
    """
    PyTorch Dataset for character-level text generation.
    Creates input-target pairs from encoded text sequences.
    
    Args:
        encoded_data: Tensor of encoded integers
        seq_length: Length of input sequences (context window)
    """
    
    def __init__(self, encoded_data, seq_length=100):
        """
        Initialize dataset with encoded data and sequence length.
        
        Args:
            encoded_data: 1D tensor of encoded integers
            seq_length: Length of input sequences
        """
        self.encoded_data = encoded_data
        self.seq_length = seq_length
        
        # Validate data
        if len(self.encoded_data) <= self.seq_length:
            raise ValueError(
                f"Encoded data ({len(self.encoded_data)}) must be longer than "
                f"sequence length ({self.seq_length})"
            )
    
    def __len__(self):
        """Return number of valid sequences."""
        # We need seq_length + 1 characters to create one input-target pair
        return len(self.encoded_data) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Get input-target pair at index.
        
        Args:
            idx: Index of sequence
            
        Returns:
            Tuple of (input_sequence, target_value)
                - input_sequence: Tensor of shape (seq_length,)
                - target_value: Integer label (next character)
        """
        input_seq = self.encoded_data[idx:idx + self.seq_length]
        target = self.encoded_data[idx + self.seq_length]
        
        return input_seq, target


def create_sequences(encoded_data, seq_length=100):
    """
    Create sequence dataset from encoded data.
    
    Args:
        encoded_data: 1D tensor of encoded integers
        seq_length: Length of input sequences
        
    Returns:
        CharacterDataset instance
    """
    dataset = CharacterDataset(encoded_data, seq_length=seq_length)
    print(f"Created dataset with {len(dataset)} sequences")
    print(f"Sequence length: {seq_length}")
    return dataset


def train_test_split(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: PyTorch Dataset instance
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_dataset)} sequences ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} sequences ({val_ratio*100:.1f}%)")
    print(f"  Test:       {len(test_dataset)} sequences ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loader(dataset, batch_size=64, shuffle=True, num_workers=0):
    """
    Create PyTorch DataLoader from dataset.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_batch(dataloader):
    """
    Get a single batch from dataloader for inspection.
    
    Args:
        dataloader: PyTorch DataLoader instance
        
    Returns:
        Tuple of (input_batch, target_batch)
    """
    for input_batch, target_batch in dataloader:
        return input_batch, target_batch


def sample_batch_info(input_batch, target_batch, int_to_char=None):
    """
    Print information about a batch for verification.
    
    Args:
        input_batch: Tensor of shape (batch_size, seq_length)
        target_batch: Tensor of shape (batch_size,)
        int_to_char: Optional dictionary for decoding integers to characters
    """
    batch_size, seq_length = input_batch.shape
    
    print("\n" + "="*60)
    print("BATCH INFORMATION")
    print("="*60)
    print(f"Input batch shape: {input_batch.shape} (batch_size={batch_size}, seq_length={seq_length})")
    print(f"Target batch shape: {target_batch.shape}")
    print(f"Input dtype: {input_batch.dtype}, Target dtype: {target_batch.dtype}")
    print(f"Input range: [{input_batch.min().item()}, {input_batch.max().item()}]")
    print(f"Target range: [{target_batch.min().item()}, {target_batch.max().item()}]")
    
    # Show first sample in batch
    print(f"\nFirst sample in batch:")
    print(f"  Input sequence: {input_batch[0].tolist()}")
    print(f"  Target value: {target_batch[0].item()}")
    
    # Decode if mapping provided
    if int_to_char is not None:
        try:
            from src.utils import decode_integers
            decoded_input = decode_integers(input_batch[0], int_to_char)
            decoded_target = int_to_char[target_batch[0].item()]
            print(f"\nDecoded first sample:")
            print(f"  Input text (first 50 chars): {decoded_input[:50]}...")
            print(f"  Target character: '{decoded_target}'")
        except Exception as e:
            # If decode fails (NumPy compatibility), still show the encoding worked
            print(f"\nNote: Could not decode (NumPy compatibility): {type(e).__name__}")
    
    print("="*60 + "\n")
