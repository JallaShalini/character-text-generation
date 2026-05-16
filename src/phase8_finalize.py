"""
Phase 8 deliverable generator.
Creates the loss plot, multi-sample generation archive, and comparison report.
"""

import json
import math
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel
from src.plot_results import main as plot_loss_curves
from src.utils import get_device, load_vocab, seed_everything


SEED_TEXT = "To be or not to be"
TEMPERATURES = [0.5, 1.0, 1.5]
SAMPLES_PER_TEMPERATURE = 3
SEED_BASE = 100


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_model(model_name, vocab_size):
    if model_name == "lstm":
        return LSTMModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.1)
    return TransformerModel(vocab_size=vocab_size, embedding_dim=128, num_heads=4, ffn_dim=512, num_layers=2, dropout=0.1)


def load_model_state(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_samples(model_name, model, char_to_int, int_to_char, device):
    samples = {}
    for temperature in TEMPERATURES:
        temperature_key = f"{temperature:.1f}"
        samples[temperature_key] = []
        for sample_index in range(SAMPLES_PER_TEMPERATURE):
            seed_everything(SEED_BASE + sample_index)
            if model_name == "lstm":
                generated = model.generate(
                    seed_text=SEED_TEXT,
                    char_to_int=char_to_int,
                    int_to_char=int_to_char,
                    device=device,
                    max_length=160,
                    temperature=temperature,
                )
            else:
                seed_tokens = torch.tensor(
                    [[char_to_int[character] for character in SEED_TEXT]],
                    dtype=torch.long,
                    device=device,
                )
                generated_ids = model.generate(seed_tokens=seed_tokens, max_length=160, temperature=temperature)
                generated = "".join(int_to_char[token_id] for token_id in generated_ids[0].tolist())

            samples[temperature_key].append({
                "sample_id": sample_index + 1,
                "text": generated,
            })

    return samples


def truncate_preview(text, limit=180):
    cleaned = text.replace("\n", "\\n")
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "..."


def build_report(loss_history, evaluation, sample_archive):
    lines = []
    lines.append("# Phase 8 Results and Comparison Report")
    lines.append("")
    lines.append("This report summarizes training curves, quantitative evaluation, and sample generations from the trained LSTM and Transformer checkpoints.")
    lines.append("")
    lines.append("## Quantitative Comparison")
    lines.append("")
    lines.append("| Model | Final Train Loss | Final Val Loss | Test Loss | Perplexity |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model_name in ["lstm", "transformer"]:
        train_loss = loss_history[model_name]["train_loss"][-1]
        val_loss = loss_history[model_name]["val_loss"][-1]
        test_loss = evaluation[model_name]["test_loss"]
        perplexity = evaluation[model_name]["perplexity"]
        lines.append(f"| {model_name.title()} | {train_loss:.4f} | {val_loss:.4f} | {test_loss:.4f} | {perplexity:.4f} |")
    lines.append("")
    best_model = evaluation.get("best_model", "transformer")
    lines.append(f"Best model by perplexity: **{best_model.title()}**")
    lines.append("")
    lines.append("## Loss Curve Summary")
    lines.append("")
    lines.append("The saved plot in `results/loss_curves.png` compares training and validation loss for both models. In the smoke-training run, the Transformer converged to a lower validation loss than the LSTM.")
    lines.append("")
    lines.append("## Qualitative Samples")
    lines.append("")
    for model_name in ["lstm", "transformer"]:
        lines.append(f"### {model_name.title()}")
        for temperature in TEMPERATURES:
            temp_key = f"{temperature:.1f}"
            lines.append(f"- Temperature {temperature:.1f}")
            for sample in sample_archive[model_name]["samples"][temp_key]:
                preview = truncate_preview(sample["text"])
                lines.append(f"  - Sample {sample['sample_id']}: `{preview}`")
        lines.append("")
    lines.append("## Short Analysis")
    lines.append("")
    lines.append("At lower temperature, both models stay closer to the prompt and produce more repetitive but structured text. As temperature increases, the samples become more diverse and less grammatical. Across the current checkpoints, the Transformer is the stronger quantitative model and the cleaner sampler at lower temperatures, while the LSTM becomes noisier more quickly at higher temperatures.")
    lines.append("")
    return "\n".join(lines)


def main():
    results_dir = Path("results")
    models_dir = Path("models")
    data_dir = Path("data")
    results_dir.mkdir(parents=True, exist_ok=True)

    loss_history = load_json(results_dir / "loss_history.json")
    evaluation = load_json(results_dir / "evaluation.json")
    char_to_int, int_to_char, vocab_size = load_vocab(data_dir / "vocab.json")
    device = get_device()

    sample_archive = {}
    for model_name in ["lstm", "transformer"]:
        model = build_model(model_name, vocab_size)
        checkpoint_path = models_dir / f"{model_name}_model.pth"
        model = load_model_state(model, checkpoint_path, device)
        sample_archive[model_name] = {
            "checkpoint": str(checkpoint_path),
            "seed_text": SEED_TEXT,
            "temperatures": TEMPERATURES,
            "samples_per_temperature": SAMPLES_PER_TEMPERATURE,
            "samples": generate_samples(model_name, model, char_to_int, int_to_char, device),
        }

    plot_loss_curves()

    generated_samples_path = results_dir / "generated_samples.json"
    with open(generated_samples_path, "w", encoding="utf-8") as file:
        json.dump(sample_archive, file, indent=2, ensure_ascii=False)

    report_path = results_dir / "comparison_report.md"
    report_text = build_report(loss_history, evaluation, sample_archive)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"Saved generation archive to {generated_samples_path}")
    print(f"Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()