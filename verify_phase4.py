"""
Verification script for Phase 4: LSTM Model Implementation
Tests model instantiation, forward pass, output shapes, and hidden state handling.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.utils import load_vocab, load_encoded_data, seed_everything, print_data_info
from src.dataset import create_sequences, train_test_split, create_data_loader, get_batch
from src.model_lstm import LSTMModel, create_lstm_model


def verify_phase_4():
    """Execute Phase 4 verification tests."""
    
    print("\n" + "="*60)
    print("PHASE 4: LSTM MODEL IMPLEMENTATION VERIFICATION")
    print("="*60)
    
    # Set seed for reproducibility
    seed_everything(seed=42)
    
    # Step 1: Load vocabulary and data
    print("\n[1/7] Loading vocabulary and encoded data...")
    try:
        char_to_int, int_to_char, vocab_size = load_vocab("data/vocab.json")
        encoded_data = load_encoded_data("data/encoded_data.pt")
        print(f"✓ Vocabulary loaded: {vocab_size} characters")
        print(f"✓ Encoded data shape: {encoded_data.shape}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False
    
    # Step 2: Create dataset and dataloader
    print("\n[2/7] Creating dataset and dataloader...")
    try:
        seq_length = 100
        batch_size = 64
        dataset = create_sequences(encoded_data, seq_length=seq_length)
        train_dataset, val_dataset, test_dataset = train_test_split(dataset)
        train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"✓ Dataset created with {len(train_dataset)} sequences")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False
    
    # Step 3: Instantiate LSTM model
    print("\n[3/7] Instantiating LSTM model...")
    try:
        embedding_dim = 128
        hidden_dim = 256
        n_layers = 2
        dropout = 0.1
        
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        print(f"✓ LSTMModel instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        return False
    
    # Step 4: Get a batch and verify shapes
    print("\n[4/7] Getting batch and verifying input shapes...")
    try:
        input_batch, target_batch = get_batch(train_loader)
        print(f"✓ Batch retrieved")
        print(f"  Input batch shape: {input_batch.shape}")
        print(f"  Target batch shape: {target_batch.shape}")
        
        assert input_batch.shape == (batch_size, seq_length), f"Input shape mismatch"
        assert target_batch.shape == (batch_size,), f"Target shape mismatch"
        print(f"✓ Input shapes verified")
    except Exception as e:
        print(f"✗ Failed to get batch: {e}")
        return False
    
    # Step 5: Test forward pass without hidden state
    print("\n[5/7] Testing forward pass without initial hidden state...")
    try:
        logits, hidden = model(input_batch)
        
        # Verify output shapes
        expected_logits_shape = (batch_size, seq_length, vocab_size)
        assert logits.shape == expected_logits_shape, \
            f"Logits shape {logits.shape} != expected {expected_logits_shape}"
        
        # Verify hidden state shapes
        h_n, c_n = hidden
        expected_hidden_shape = (n_layers, batch_size, hidden_dim)
        assert h_n.shape == expected_hidden_shape, \
            f"Hidden state shape {h_n.shape} != expected {expected_hidden_shape}"
        assert c_n.shape == expected_hidden_shape, \
            f"Cell state shape {c_n.shape} != expected {expected_hidden_shape}"
        
        print(f"✓ Forward pass successful (no initial hidden state)")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hidden state shape: {h_n.shape}")
        print(f"  Cell state shape: {c_n.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Step 6: Test forward pass with provided hidden state
    print("\n[6/7] Testing forward pass with initial hidden state...")
    try:
        # Initialize hidden states
        h_0, c_0 = model.init_hidden(batch_size, input_batch.device)
        
        # Forward pass with provided hidden state
        logits2, hidden2 = model(input_batch, (h_0, c_0))
        
        assert logits2.shape == (batch_size, seq_length, vocab_size), \
            f"Logits shape mismatch with initial hidden state"
        
        h_n2, c_n2 = hidden2
        assert h_n2.shape == (n_layers, batch_size, hidden_dim), \
            f"Hidden state shape mismatch with initial hidden state"
        
        print(f"✓ Forward pass with initial hidden state successful")
        print(f"  Logits shape: {logits2.shape}")
        print(f"  Output hidden state shape: {h_n2.shape}")
    except Exception as e:
        print(f"✗ Forward pass with hidden state failed: {e}")
        return False
    
    # Step 7: Verify output logits are valid predictions
    print("\n[7/7] Verifying output logits for next-character prediction...")
    try:
        # Check logits statistics
        print(f"  Logits shape: {logits.shape} (batch_size, seq_length, vocab_size)")
        print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        print(f"  Logits mean: {logits.mean().item():.4f}")
        print(f"  Logits std: {logits.std().item():.4f}")
        
        # Convert logits to probabilities for first sample
        probs = torch.softmax(logits[0], dim=-1)  # Shape: (seq_length, vocab_size)
        
        # Verify probabilities are valid
        assert (probs >= 0).all(), "Negative probabilities detected"
        assert (probs <= 1).all(), "Probabilities > 1 detected"
        
        # Check that probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Probabilities don't sum to 1"
        
        # Get most likely next character for first position
        best_next_char_idx = logits[0, 0, :].argmax().item()
        best_next_char = int_to_char[best_next_char_idx]
        
        print(f"✓ Output logits verified")
        print(f"  Probability range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        print(f"  Most likely next char (position 0): '{best_next_char}' (idx={best_next_char_idx})")
    except Exception as e:
        print(f"✗ Logits verification failed: {e}")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("✓ PHASE 4 VERIFICATION COMPLETE!")
    print("="*60)
    print(f"\nModel Summary:")
    print(f"  Architecture: Embedding -> {n_layers}×LSTM -> Linear")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"\nInput/Output Shapes (for training):")
    print(f"  Batch input: {input_batch.shape}")
    print(f"  Model output: {logits.shape}")
    print(f"  Batch targets: {target_batch.shape}")
    print(f"\n✓ LSTM Model is ready for training!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = verify_phase_4()
    sys.exit(0 if success else 1)
