"""
Text generation script for trained character-level models.
Supports both LSTM and Transformer checkpoints with temperature scaling.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel
from src.utils import get_device, load_vocab, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained character model.")
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed-text", type=str, default="To be or not to be")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatures", type=float, nargs="*", default=None)
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/generated_samples.json")
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


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def sample_with_model(model, model_name, seed_text, char_to_int, int_to_char, temperature, length, device):
    if model_name == "lstm":
        return model.generate(
            seed_text=seed_text,
            char_to_int=char_to_int,
            int_to_char=int_to_char,
            device=device,
            max_length=length,
            temperature=temperature,
        )

    seed_tokens = torch.tensor(
        [[char_to_int[character] for character in seed_text]],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(seed_tokens=seed_tokens, max_length=length, temperature=temperature)
    generated_ids = generated_ids[0].tolist()
    return "".join(int_to_char[token_id] for token_id in generated_ids)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    data_dir = Path("data")
    models_dir = Path("models")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    char_to_int, int_to_char, vocab_size = load_vocab(data_dir / "vocab.json")
    model = build_model(args.model, vocab_size, args, device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = models_dir / f"{args.model}_model.pth"
    model = load_checkpoint(model, checkpoint_path, device)

    temperatures = args.temperatures if args.temperatures else [args.temperature]
    if args.temperatures is None and args.temperature == 1.0:
        temperatures = [0.5, 1.0, 1.5]

    generations = {}

    print("\n" + "=" * 60)
    print(f"TEXT GENERATION: {args.model.upper()}")
    print("=" * 60)
    print(f"Seed text: {args.seed_text!r}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generation length: {args.length}")

    for temperature in temperatures:
        generated_text = sample_with_model(
            model=model,
            model_name=args.model,
            seed_text=args.seed_text,
            char_to_int=char_to_int,
            int_to_char=int_to_char,
            temperature=temperature,
            length=args.length,
            device=device,
        )
        generations[str(temperature)] = generated_text
        print("\n" + "-" * 60)
        print(f"Temperature: {temperature}")
        print(generated_text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "seed_text": args.seed_text,
        "length": args.length,
        "temperatures": temperatures,
        "generations": generations,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Saved generations to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
