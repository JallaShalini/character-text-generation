"""
Quantitative evaluation script for trained character models.
Computes test-set cross-entropy loss and perplexity for each model.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import create_data_loader, create_sequences, train_test_split
from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel
from src.utils import get_device, load_encoded_data, load_vocab, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained character models on the test set.")
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--output", type=str, default="results/evaluation.json")
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
        checkpoint = Path("models/lstm_model.pth")
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        checkpoint = Path("models/transformer_model.pth")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device), checkpoint


def evaluate_model(model, dataloader, device, max_batches=0):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            next_token_logits = logits[:, -1, :]
            loss = criterion(next_token_logits, targets)

            total_loss += loss.item()
            total_examples += targets.size(0)

            if max_batches > 0 and batch_index + 1 >= max_batches:
                break

    if total_examples == 0:
        raise RuntimeError("No evaluation examples were processed.")

    average_loss = total_loss / total_examples
    perplexity = math.exp(min(average_loss, 20.0))
    return average_loss, perplexity


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, vocab_size = load_vocab(data_dir / "vocab.json")
    encoded_data = load_encoded_data(data_dir / "encoded_data.pt")
    dataset = create_sequences(encoded_data, seq_length=args.seq_length)
    _, _, test_dataset = train_test_split(dataset)
    test_loader = create_data_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_names = ["lstm", "transformer"] if args.model == "both" else [args.model]
    report = {}

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    for model_name in model_names:
        model, checkpoint_path = build_model(model_name, vocab_size, args, device)
        average_loss, perplexity = evaluate_model(
            model,
            test_loader,
            device,
            max_batches=args.max_batches,
        )

        report[model_name] = {
            "test_loss": average_loss,
            "perplexity": perplexity,
            "checkpoint": str(checkpoint_path),
        }

        print(f"{model_name.upper()}: test_loss={average_loss:.4f} | perplexity={perplexity:.4f}")

    if len(model_names) == 2:
        best_model = min(report, key=lambda name: report[name]["perplexity"])
        report["best_model"] = best_model
        print(f"Best model by perplexity: {best_model}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved evaluation report to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
