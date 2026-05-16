"""
Data preparation and preprocessing pipeline for character-level text generation.
Handles text loading, vocabulary building, integer encoding, and sequence creation.
"""

import json
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def load_text(filepath):
    """Load raw text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded text from {filepath}: {len(text)} characters")
    return text


def build_vocab(text):
    """Build vocabulary from text - collect all unique characters."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Sample characters: {chars[:20]}")
    return chars


def create_mappings(vocab):
    """Create bidirectional character-to-integer and integer-to-character mappings."""
    char_to_int = {char: idx for idx, char in enumerate(vocab)}
    int_to_char = {idx: char for idx, char in enumerate(vocab)}
    return char_to_int, int_to_char


def encode_data(text, char_to_int):
    """Encode entire text to integer sequence."""
    print("Encoding text to integers...")
    encoded = torch.tensor([char_to_int[char] for char in tqdm(text, desc="Encoding")], dtype=torch.long)
    print(f"Encoded data shape: {encoded.shape}")
    return encoded


def create_sequences(encoded_data, seq_length):
    """Create input-target pairs from encoded data."""
    print(f"Creating sequences of length {seq_length}...")
    sequences = []
    for i in tqdm(range(0, len(encoded_data) - seq_length), desc="Creating sequences"):
        input_seq = encoded_data[i:i+seq_length]
        target = encoded_data[i+seq_length]
        sequences.append((input_seq, target))
    
    print(f"Created {len(sequences)} training sequences")
    return sequences


def save_artifacts(char_to_int, int_to_char, encoded_data, data_dir):
    """Save all preprocessed data artifacts."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vocabulary mappings
    vocab_dict = {
        "char_to_int": char_to_int,
        "int_to_char": {str(k): v for k, v in int_to_char.items()},  # JSON requires string keys
        "vocab_size": len(char_to_int)
    }
    vocab_path = data_dir / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save mappings for reference
    mappings_dict = {
        "total_chars": len(char_to_int),
        "vocab_size": len(char_to_int),
        "char_to_int_sample": {k: char_to_int[k] for k in sorted(char_to_int.keys())[:10]}
    }
    mappings_path = data_dir / "mappings.json"
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(mappings_dict, f, indent=2)
    print(f"Saved mappings to {mappings_path}")
    
    # Save encoded data as PyTorch tensor
    encoded_path = data_dir / "encoded_data.pt"
    torch.save(encoded_data, encoded_path)
    print(f"Saved encoded data to {encoded_path} (size: {encoded_data.shape})")
    
    return vocab_path, mappings_path, encoded_path


def load_and_verify(data_dir):
    """Verify that saved artifacts can be loaded correctly."""
    print("\n" + "="*60)
    print("VERIFICATION: Loading saved artifacts...")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Load vocabulary
    with open(data_dir / "vocab.json", 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    print(f"✓ Loaded vocabulary: {vocab_dict['vocab_size']} characters")
    
    # Load mappings
    with open(data_dir / "mappings.json", 'r', encoding='utf-8') as f:
        mappings_dict = json.load(f)
    print(f"✓ Loaded mappings: {mappings_dict['vocab_size']} total chars")
    
    # Load encoded data
    encoded_data = torch.load(data_dir / "encoded_data.pt")
    print(f"✓ Loaded encoded data: shape {encoded_data.shape}, dtype {encoded_data.dtype}")
    
    # Verify data integrity
    print(f"✓ Data range: [{encoded_data.min().item()}, {encoded_data.max().item()}]")
    print(f"✓ Expected range: [0, {vocab_dict['vocab_size']-1}]")
    
    assert encoded_data.min().item() >= 0, "Encoded data contains negative values!"
    assert encoded_data.max().item() < vocab_dict['vocab_size'], "Encoded data exceeds vocab size!"
    
    print("\n✓ ALL VERIFICATION CHECKS PASSED!")
    return vocab_dict, encoded_data


def main():
    # Paths
    input_path = Path("input/shakespeare.txt")
    data_dir = Path("data")
    
    # Check input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Shakespeare corpus not found at {input_path}")
    
    print("="*60)
    print("PHASE 2: DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load text
    text = load_text(input_path)
    
    # Step 2: Build vocabulary
    vocab = build_vocab(text)
    
    # Step 3: Create mappings
    char_to_int, int_to_char = create_mappings(vocab)
    
    # Step 4: Encode data
    encoded_data = encode_data(text, char_to_int)
    
    # Step 5: Save artifacts
    save_artifacts(char_to_int, int_to_char, encoded_data, data_dir)
    
    # Step 6: Verify by loading
    vocab_dict, loaded_data = load_and_verify(data_dir)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    print(f"Generated artifacts in {data_dir}:")
    print(f"  - encoded_data.pt: {loaded_data.shape}")
    print(f"  - vocab.json: {len(char_to_int)} unique characters")
    print(f"  - mappings.json: metadata and sample mappings")


if __name__ == "__main__":
    main()
