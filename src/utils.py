"""
Utility functions for character mapping, tensor operations, and data handling.
Centralized helpers for both training and generation scripts.
"""

import json
import torch
import random
import numpy as np
from pathlib import Path


def load_vocab(vocab_path="data/vocab.json"):
    """
    Load vocabulary mappings from saved JSON file.
    
    Args:
        vocab_path: Path to vocab.json file
        
    Returns:
        char_to_int: Dictionary mapping characters to integers
        int_to_char: Dictionary mapping integers to characters
        vocab_size: Number of unique characters
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    
    char_to_int = vocab_dict['char_to_int']
    # Convert string keys back to integers for int_to_char
    int_to_char = {int(k): v for k, v in vocab_dict['int_to_char'].items()}
    vocab_size = vocab_dict['vocab_size']
    
    return char_to_int, int_to_char, vocab_size


def encode_text(text, char_to_int):
    """
    Encode raw text to integer sequence using character mapping.
    
    Args:
        text: Raw text string
        char_to_int: Dictionary mapping characters to integers
        
    Returns:
        Encoded tensor of integers
    """
    encoded = torch.tensor(
        [char_to_int[char] for char in text],
        dtype=torch.long
    )
    return encoded


def decode_integers(integers, int_to_char):
    """
    Decode integer sequence back to text.
    
    Args:
        integers: Tensor or list of integers
        int_to_char: Dictionary mapping integers to characters
        
    Returns:
        Decoded text string
    """
    if isinstance(integers, torch.Tensor):
        integers = integers.cpu().numpy().tolist()
    elif not isinstance(integers, list):
        integers = list(integers)
    
    text = ''.join(int_to_char[int(idx)] for idx in integers)
    return text


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def get_device():
    """
    Get the device to use for training (GPU if available, else CPU).
    
    Returns:
        torch.device: CPU or CUDA device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_encoded_data(encoded_path="data/encoded_data.pt"):
    """
    Load pre-encoded integer tensor from file.
    
    Args:
        encoded_path: Path to encoded_data.pt file
        
    Returns:
        Encoded tensor
    """
    encoded_path = Path(encoded_path)
    if not encoded_path.exists():
        raise FileNotFoundError(f"Encoded data file not found at {encoded_path}")
    
    encoded_data = torch.load(encoded_path)
    return encoded_data


def move_to_device(batch, device):
    """
    Move batch data to specified device.
    
    Args:
        batch: Tuple of (input_ids, target_ids) or single tensor
        device: Target device (CPU or GPU)
        
    Returns:
        Batch moved to device
    """
    if isinstance(batch, (list, tuple)):
        return tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def print_data_info(encoded_data, vocab_size):
    """
    Print summary statistics about the data.
    
    Args:
        encoded_data: Encoded integer tensor
        vocab_size: Size of vocabulary
    """
    print("\n" + "="*60)
    print("DATA INFORMATION")
    print("="*60)
    print(f"Encoded data shape: {encoded_data.shape}")
    print(f"Encoded data dtype: {encoded_data.dtype}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Data range: [{encoded_data.min().item()}, {encoded_data.max().item()}]")
    print(f"Memory size: {encoded_data.element_size() * encoded_data.nelement() / (1024**2):.2f} MB")
    print("="*60 + "\n")
