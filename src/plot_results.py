"""
Plot training and validation loss curves from saved training history.
Outputs results/loss_curves.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_loss_history(history_path):
    with open(history_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    results_dir = Path("results")
    history_path = results_dir / "loss_history.json"
    output_path = results_dir / "loss_curves.png"

    if not history_path.exists():
        raise FileNotFoundError(f"Loss history not found: {history_path}")

    history = load_loss_history(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = [
        ("lstm", "LSTM", "#1f77b4"),
        ("transformer", "Transformer", "#d62728"),
    ]

    for axis, (model_key, model_label, color) in zip(axes, models):
        model_history = history.get(model_key)
        if not model_history:
            raise KeyError(f"Missing history for {model_key}")

        train_loss = model_history["train_loss"]
        val_loss = model_history["val_loss"]
        epochs = list(range(1, len(train_loss) + 1))

        axis.plot(epochs, train_loss, marker="o", color=color, label="Train loss")
        axis.plot(epochs, val_loss, marker="s", linestyle="--", color="#444444", label="Val loss")
        axis.set_title(model_label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("Training Curves for Character-Level Text Generation Models", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()