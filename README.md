# Character Text Generation

Character-level text generation with two baseline models: an LSTM and a mini-Transformer. The project is designed to run end-to-end in Docker and produces saved checkpoints, evaluation metrics, samples, and plots.

## Project Overview

The pipeline follows these stages:

1. Build the project skeleton and Docker setup.
2. Download and preprocess the Tiny Shakespeare corpus.
3. Implement a reusable dataset/batching layer.
4. Train and evaluate an LSTM baseline.
5. Train and evaluate a Transformer baseline.
6. Use a shared training script for both models.
7. Generate samples and compute perplexity.
8. Produce plots, JSON sample archives, and a comparison report.
9. Validate the full workflow inside Docker.

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- tqdm
- Docker and Docker Compose

## Project Structure

```text
character-text-generation/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── input/
├── data/
├── models/
├── results/
└── src/
    ├── prepare_data.py
    ├── dataset.py
    ├── model_lstm.py
    ├── model_transformer.py
    ├── train.py
    ├── generate.py
    ├── evaluate.py
    ├── plot_results.py
    ├── phase8_finalize.py
    └── utils.py
```

## Setup Instructions

### Local environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Docker

```bash
docker compose build
```

## Required Execution Order

### 1. Prepare data

```bash
python src/prepare_data.py
```

### 2. Train LSTM

```bash
python src/train.py --model lstm
```

### 3. Train Transformer

```bash
python src/train.py --model transformer
```

### 4. Generate text

```bash
python src/generate.py --model lstm
```

### 5. Plot results

```bash
python src/plot_results.py
```

### 6. Evaluate models

```bash
python src/evaluate.py
```

### 7. Finalize outputs

```bash
python src/phase8_finalize.py
```

## Docker Commands

Build:

```bash
docker compose build
```

Train LSTM:

```bash
docker compose run --rm app python src/train.py --model lstm
```

Train Transformer:

```bash
docker compose run --rm app python src/train.py --model transformer
```

Generate text:

```bash
docker compose run --rm app python src/generate.py --model lstm
```

Run evaluation:

```bash
docker compose run --rm app python src/evaluate.py
```

Create final plot/report/sample outputs:

```bash
docker compose run --rm app python src/phase8_finalize.py
```

## Results

The project writes its outputs to these folders:

- `data/encoded_data.pt`
- `data/vocab.json`
- `data/mappings.json`
- `models/lstm_model.pth`
- `models/transformer_model.pth`
- `results/loss_history.json`
- `results/evaluation.json`
- `results/generated_samples.json`
- `results/loss_curves.png`
- `results/comparison_report.md`

## Model Comparison

Current validated metrics:

| Model | Test Loss | Perplexity |
| --- | ---: | ---: |
| LSTM | 3.3454 | 28.3716 |
| Transformer | 3.1143 | 22.5187 |

The Transformer performs better on perplexity in the current smoke-training run.

## Notes

The repository is already fully implemented and validated. The README now reflects the complete submission and execution flow.
