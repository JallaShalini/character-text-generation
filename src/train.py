"""
Shared training pipeline for the character-level text generation project.
Supports both the LSTM and Transformer models via a single command-line flag.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import create_data_loader, create_sequences, train_test_split
from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel
from src.utils import get_device, load_encoded_data, load_vocab, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LSTM or Transformer character model.")
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=25)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(model_name, vocab_size, args, device):
    if model_name == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    return model.to(device)


def compute_batch_loss(model, batch_inputs, batch_targets, criterion, model_name):
    if model_name == "lstm":
        logits, _ = model(batch_inputs)
    else:
        logits, _ = model(batch_inputs)

    next_token_logits = logits[:, -1, :]
    loss = criterion(next_token_logits, batch_targets)
    return loss


def evaluate(model, dataloader, criterion, device, model_name, max_batches=None):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = compute_batch_loss(model, inputs, targets, criterion, model_name)
            losses.append(loss.item())

            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    return sum(losses) / max(len(losses), 1)


def load_or_init_history(results_dir, model_name):
    history_path = results_dir / "loss_history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = {}

    history.setdefault(model_name, {"train_loss": [], "val_loss": []})
    return history_path, history


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    data_dir = Path("data")
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, vocab_size = load_vocab(data_dir / "vocab.json")
    encoded_data = load_encoded_data(data_dir / "encoded_data.pt")

    dataset = create_sequences(encoded_data, seq_length=args.seq_length)
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)

    train_loader = create_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, vocab_size, args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history_path, history = load_or_init_history(results_dir, args.model)
    train_history = history[args.model]["train_loss"]
    val_history = history[args.model]["val_loss"]

    print("\n" + "=" * 60)
    print(f"TRAINING STARTED: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Max batches per epoch: {args.max_batches if args.max_batches > 0 else 'all'}")

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []

        for batch_index, (inputs, targets) in enumerate(train_loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_batch_loss(model, inputs, targets, criterion, args.model)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            batch_losses.append(loss.item())

            if (batch_index + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_index + 1} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = sum(batch_losses) / max(len(batch_losses), 1)
        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.model,
            max_batches=args.max_batches if args.max_batches > 0 else None,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} complete | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    model_path = models_dir / f"{args.model}_model.pth"
    torch.save(model.state_dict(), model_path)

    history[args.model]["train_loss"] = train_history
    history[args.model]["val_loss"] = val_history
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved model weights to: {model_path}")
    print(f"Saved loss history to: {history_path}")
    print(f"Final train loss: {train_history[-1]:.4f}")
    print(f"Final val loss: {val_history[-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()