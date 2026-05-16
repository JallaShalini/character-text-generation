"""
Verification script for Phase 5: Mini-Transformer implementation.
Tests instantiation, forward pass, and output tensor shapes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from src.dataset import create_data_loader, create_sequences, get_batch, train_test_split
from src.model_transformer import TransformerModel
from src.utils import load_encoded_data, load_vocab, seed_everything


def verify_phase_5():
    print("\n" + "=" * 60)
    print("PHASE 5: MINI-TRANSFORMER IMPLEMENTATION VERIFICATION")
    print("=" * 60)

    seed_everything(seed=42)

    print("\n[1/6] Loading vocabulary and encoded data...")
    try:
        _, _, vocab_size = load_vocab("data/vocab.json")
        encoded_data = load_encoded_data("data/encoded_data.pt")
        print(f"✓ Vocabulary loaded: {vocab_size} characters")
        print(f"✓ Encoded data shape: {encoded_data.shape}")
    except Exception as error:
        print(f"✗ Failed to load data: {error}")
        return False

    print("\n[2/6] Building dataset and data loader...")
    try:
        dataset = create_sequences(encoded_data, seq_length=100)
        train_dataset, _, _ = train_test_split(dataset)
        train_loader = create_data_loader(train_dataset, batch_size=64, shuffle=True)
        print(f"✓ Dataset ready with {len(train_dataset)} training sequences")
    except Exception as error:
        print(f"✗ Failed to build dataset: {error}")
        return False

    print("\n[3/6] Instantiating Transformer model...")
    try:
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_heads=4,
            ffn_dim=512,
            num_layers=2,
            dropout=0.1,
        )
        print("✓ TransformerModel instantiated successfully")
    except Exception as error:
        print(f"✗ Failed to instantiate model: {error}")
        return False

    print("\n[4/6] Retrieving a batch...")
    try:
        input_batch, target_batch = get_batch(train_loader)
        assert input_batch.shape == (64, 100), f"Unexpected input shape: {input_batch.shape}"
        assert target_batch.shape == (64,), f"Unexpected target shape: {target_batch.shape}"
        print(f"✓ Input batch shape: {input_batch.shape}")
        print(f"✓ Target batch shape: {target_batch.shape}")
    except Exception as error:
        print(f"✗ Failed to retrieve batch: {error}")
        return False

    print("\n[5/6] Running forward pass...")
    try:
        logits, hidden = model(input_batch)
        expected_shape = (64, 100, vocab_size)
        assert logits.shape == expected_shape, f"Logits shape {logits.shape} != {expected_shape}"
        assert hidden is None
        print(f"✓ Forward pass successful: logits shape {logits.shape}")
    except Exception as error:
        print(f"✗ Forward pass failed: {error}")
        return False

    print("\n[6/6] Validating output logits...")
    try:
        probs = torch.softmax(logits[0], dim=-1)
        assert probs.shape == (100, vocab_size)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(100), atol=1e-5)
        print(f"✓ Probability tensor shape: {probs.shape}")
        print(f"✓ Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"✓ Model parameter count: {model.count_parameters():,}")
    except Exception as error:
        print(f"✗ Output validation failed: {error}")
        return False

    print("\n" + "=" * 60)
    print("✓ PHASE 5 VERIFICATION COMPLETE!")
    print("=" * 60)
    print(f"\nInput batch shape: {input_batch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    print("\n✓ Transformer model is ready for the next phase!")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = verify_phase_5()
    sys.exit(0 if success else 1)