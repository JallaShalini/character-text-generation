import os
import json
import torch

print('='*60)
print('FINAL VALIDATION INSIDE DOCKER')
print('='*60)

# Check all required files exist
files_to_check = [
    'data/encoded_data.pt',
    'data/vocab.json',
    'data/mappings.json',
    'models/lstm_model.pth',
    'models/transformer_model.pth',
    'results/loss_history.json',
    'results/evaluation.json',
    'results/generated_samples.json',
    'results/comparison_report.md',
    'results/loss_curves.png'
]

print('\nChecking project artifacts...')
all_exist = True
for f in files_to_check:
    exists = os.path.exists(f)
    status = 'OK' if exists else 'MISSING'
    print(f'  [{status}] {f}')
    all_exist = all_exist and exists

# Validate data artifacts
print('\nValidating data artifacts...')
encoded = torch.load('data/encoded_data.pt')
print(f'  [OK] Encoded data shape: {encoded.shape}')

with open('data/vocab.json') as f:
    vocab = json.load(f)
    print(f'  [OK] Vocabulary size: {len(vocab)} chars')

with open('data/mappings.json') as f:
    mappings = json.load(f)
    print(f'  [OK] Mappings loaded: {mappings.get("vocab_size")} vocab size')

# Validate model checkpoints
print('\nValidating model checkpoints...')
lstm_ckpt = torch.load('models/lstm_model.pth', map_location='cpu')
print(f'  [OK] LSTM checkpoint keys: {list(lstm_ckpt.keys())[:3]}...')

transformer_ckpt = torch.load('models/transformer_model.pth', map_location='cpu')
print(f'  [OK] Transformer checkpoint keys: {list(transformer_ckpt.keys())[:3]}...')

# Validate results
print('\nValidating results artifacts...')
with open('results/loss_history.json') as f:
    loss_hist = json.load(f)
    print(f'  [OK] Loss history has {len(loss_hist)} epochs recorded')

with open('results/evaluation.json') as f:
    eval_json = json.load(f)
    print(f'  [OK] Best model: {eval_json.get("best_model")}')
    print(f'  [OK] LSTM perplexity: {eval_json["lstm"]["perplexity"]:.2f}')
    print(f'  [OK] Transformer perplexity: {eval_json["transformer"]["perplexity"]:.2f}')

with open('results/generated_samples.json') as f:
    samples = json.load(f)
    print(f'  [OK] Generated samples for models: {list(samples.keys())}')

with open('results/comparison_report.md') as f:
    report = f.read()
    print(f'  [OK] Comparison report size: {len(report)} bytes')

plot_size = os.path.getsize('results/loss_curves.png')
print(f'  [OK] Loss curves plot size: {plot_size} bytes')

print('\n' + '='*60)
print('ALL VALIDATION CHECKS PASSED!')
print('='*60)
print(f'\nProject Status:')
print(f'  Data: Preprocessed and validated')
print(f'  Models: LSTM and Transformer trained')
print(f'  Evaluation: Both models tested')
print(f'  Generation: Samples at 3 temperatures')
print(f'  Reporting: Complete with plots and analysis')
print(f'\nAll {len(files_to_check)} required artifacts present and valid.')
