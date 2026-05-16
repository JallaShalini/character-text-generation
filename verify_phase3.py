"""
Verification script for Phase 3: Dataset and batching utilities.
Tests data loading, sequence creation, batching, and shape validation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.utils import load_vocab, load_encoded_data, seed_everything, print_data_info, decode_integers
from src.dataset import (
    create_sequences, 
    train_test_split, 
    create_data_loader,
    get_batch,
    sample_batch_info
)


def verify_phase_3():
    """Execute Phase 3 verification tests."""
    
    print("\n" + "="*60)
    print("PHASE 3: DATASET AND BATCHING UTILITIES VERIFICATION")
    print("="*60)
    
    # Set seed for reproducibility
    seed_everything(seed=42)
    
    # Step 1: Load vocabulary
    print("\n[1/6] Loading vocabulary...")
    try:
        char_to_int, int_to_char, vocab_size = load_vocab("data/vocab.json")
        print(f"✓ Vocabulary loaded: {vocab_size} unique characters")
        print(f"  Sample mappings: {list(char_to_int.items())[:5]}")
    except Exception as e:
        print(f"✗ Failed to load vocabulary: {e}")
        return False
    
    # Step 2: Load encoded data
    print("\n[2/6] Loading encoded data...")
    try:
        encoded_data = load_encoded_data("data/encoded_data.pt")
        print_data_info(encoded_data, vocab_size)
    except Exception as e:
        print(f"✗ Failed to load encoded data: {e}")
        return False
    
    # Step 3: Create dataset with sequences
    print("[3/6] Creating dataset with sequences...")
    try:
        seq_length = 100  # From .env.example
        dataset = create_sequences(encoded_data, seq_length=seq_length)
        print(f"✓ Dataset created successfully")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False
    
    # Step 4: Split into train/val/test
    print("\n[4/6] Splitting dataset into train/val/test...")
    try:
        train_dataset, val_dataset, test_dataset = train_test_split(
            dataset, 
            train_ratio=0.8, 
            val_ratio=0.1
        )
        print(f"✓ Dataset split successfully")
    except Exception as e:
        print(f"✗ Failed to split dataset: {e}")
        return False
    
    # Step 5: Create data loaders
    print("\n[5/6] Creating data loaders...")
    try:
        batch_size = 64  # From .env.example
        train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"✓ Data loaders created successfully")
        print(f"  Batch size: {batch_size}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"✗ Failed to create data loaders: {e}")
        return False
    
    # Step 6: Sample and verify batch
    print("\n[6/6] Sampling and verifying batch shapes...")
    try:
        input_batch, target_batch = get_batch(train_loader)
        
        # Verify shapes
        expected_input_shape = (batch_size, seq_length)
        expected_target_shape = (batch_size,)
        
        assert input_batch.shape == expected_input_shape, \
            f"Input shape {input_batch.shape} != expected {expected_input_shape}"
        assert target_batch.shape == expected_target_shape, \
            f"Target shape {target_batch.shape} != expected {expected_target_shape}"
        
        # Display batch information (decode skipped due to NumPy compatibility)
        print("\n" + "="*60)
        print("BATCH INFORMATION")
        print("="*60)
        print(f"Input batch shape: {input_batch.shape} (batch_size={batch_size}, seq_length={seq_length})")
        print(f"Target batch shape: {target_batch.shape}")
        print(f"Input dtype: {input_batch.dtype}, Target dtype: {target_batch.dtype}")
        print(f"Input range: [{input_batch.min().item()}, {input_batch.max().item()}]")
        print(f"Target range: [{target_batch.min().item()}, {target_batch.max().item()}]")
        print(f"\nFirst input sequence (integer encoding): {input_batch[0].tolist()}")
        print(f"First target value: {target_batch[0].item()}")
        print("="*60 + "\n")
        
        print("✓ Batch shapes verified correctly")
        
    except Exception as e:
        print(f"✗ Failed to verify batch: {e}")
        return False
    
    # Step 7: Verify data consistency
    print("[7/7] Verifying data consistency...")
    try:
        # Check that all integers are within vocab range
        assert input_batch.min().item() >= 0, "Input contains negative values!"
        assert input_batch.max().item() < vocab_size, "Input exceeds vocab size!"
        assert target_batch.min().item() >= 0, "Target contains negative values!"
        assert target_batch.max().item() < vocab_size, "Target exceeds vocab size!"
        
        print(f"✓ Data consistency verified")
        print(f"  All values within range [0, {vocab_size-1}]")
        
    except Exception as e:
        print(f"✗ Data consistency check failed: {e}")
        return False
    
    # Success summary
    print("\n" + "="*60)
    print("✓ PHASE 3 VERIFICATION COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Vocabulary: {vocab_size} characters")
    print(f"  - Encoded data: {encoded_data.shape[0]} characters")
    print(f"  - Dataset: {len(dataset)} sequences (length={seq_length})")
    print(f"  - Train/Val/Test split: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input shape per batch: {input_batch.shape}")
    print(f"  - Target shape per batch: {target_batch.shape}")
    print(f"\n✓ Dataset utilities are ready for training!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = verify_phase_3()
    sys.exit(0 if success else 1)
