"""
LSTM Model for character-level text generation.
Compact CPU-friendly architecture with embedding, LSTM layers, and output projection.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based character-level text generation model.
    
    Architecture:
        1. Embedding layer: converts character integers to dense vectors
        2. LSTM layers: captures sequential dependencies
        3. Output projection: maps LSTM output to vocabulary logits
    
    Args:
        vocab_size (int): Number of unique characters in vocabulary
        embedding_dim (int): Dimension of character embeddings
        hidden_dim (int): Dimension of LSTM hidden state
        n_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout probability for regularization (default 0.1)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.1):
        """Initialize LSTM model layers."""
        super(LSTMModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Embedding layer: maps integer tokens to continuous vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers: stack of LSTM cells
        # dropout is applied between LSTM layers (only effective with n_layers > 1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,  # Input/output shape: (batch, seq, features)
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Output projection: maps LSTM hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Optional: layer normalization before output (can help with training)
        # self.ln = nn.LayerNorm(hidden_dim)
        
        print(f"LSTMModel initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  LSTM layers: {n_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden and cell states for LSTM.
        
        Args:
            batch_size (int): Number of sequences in batch
            device (torch.device): Device to place tensors on (CPU or GPU)
            
        Returns:
            Tuple of (h_0, c_0):
                - h_0: Initial hidden state, shape (n_layers, batch_size, hidden_dim)
                - c_0: Initial cell state, shape (n_layers, batch_size, hidden_dim)
        """
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return h_0, c_0
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length)
                              Contains integer-encoded characters
            hidden (tuple, optional): Tuple of (h_0, c_0) initial states.
                                     If None, initialized with zeros.
                                     
        Returns:
            Tuple of (logits, hidden):
                - logits: Output logits of shape (batch_size, seq_length, vocab_size)
                          Raw scores for each character prediction at each position
                - hidden: Tuple of (h_n, c_n) final states for next sequence
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM forward pass
        # lstm_out: (batch_size, seq_length, hidden_dim)
        # hidden: tuple of (h_n, c_n)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Output projection: (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length, vocab_size)
        logits = self.fc(lstm_out)
        
        return logits, hidden
    
    def generate(self, seed_text, char_to_int, int_to_char, device, 
                 max_length=100, temperature=1.0):
        """
        Generate text starting from seed text.
        
        Args:
            seed_text (str): Initial text to start generation
            char_to_int (dict): Character to integer mapping
            int_to_char (dict): Integer to character mapping
            device (torch.device): Device to run model on
            max_length (int): Maximum characters to generate
            temperature (float): Sampling temperature (>1: more random, <1: more deterministic)
            
        Returns:
            Generated text string
        """
        self.eval()  # Set to evaluation mode
        
        # Encode seed text
        encoded = torch.tensor(
            [char_to_int[ch] for ch in seed_text],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)  # Add batch dimension
        
        generated = list(seed_text)
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.forward(encoded, hidden)
                
                # Get logits for next character (last position in sequence)
                next_logits = logits[0, -1, :] / temperature
                
                # Sample next character
                probs = torch.softmax(next_logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = int_to_char[next_char_idx]
                
                generated.append(next_char)
                
                # Update input for next iteration (use last generated character)
                encoded = torch.tensor(
                    [[next_char_idx]],
                    dtype=torch.long,
                    device=device
                )
        
        return ''.join(generated)


def create_lstm_model(vocab_size, embedding_dim=128, hidden_dim=256, 
                      n_layers=2, dropout=0.1):
    """
    Factory function to create LSTM model with default parameters.
    
    Args:
        vocab_size (int): Vocabulary size
        embedding_dim (int): Embedding dimension (default 128)
        hidden_dim (int): Hidden state dimension (default 256)
        n_layers (int): Number of LSTM layers (default 2)
        dropout (float): Dropout rate (default 0.1)
        
    Returns:
        LSTMModel instance
    """
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
    return model
