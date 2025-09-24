# SLM – Small Language Model

A custom transformer-based language model built from scratch in PyTorch.

## Architecture

- **Parameters**: ~30M
- **Layers**: 6 transformer blocks
- **Attention**: 6-head causal self-attention with Flash Attention
- **Normalization**: RMSNorm (LLaMA-style)
- **FFN**: SwiGLU activation
- **Tokenizer**: GPT-2 BPE (tiktoken, 50,257 vocab)
- **Context Window**: 256 tokens

## Project Structure

```
├── config.py          # Hyperparameters
├── model.py           # Transformer architecture (RMSNorm, CausalSelfAttention, SwiGLU, SmallLanguageModel)
├── prepare_data.py    # Downloads & tokenizes corpus into binary streams
├── dataset.py         # Memory-mapped dataset with strided sampling
├── train.py           # Training loop (Adam, cross-entropy, gradient clipping)
├── generate.py        # CLI text generation
├── app.py             # Flask web interface
├── templates/         # HTML
├── static/            # CSS, JS
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

## Usage

**Prepare data:**
```bash
python prepare_data.py
```

**Train:**
```bash
python train.py
```

**Generate (CLI):**
```bash
python generate.py --prompt "Once upon a time" --tokens 100
```

**Web UI:**
```bash
python app.py
# Open http://localhost:5000
```

## Training Details

- **Corpus**: 1.8M+ tokens from classic literature (Project Gutenberg)
- **Optimizer**: Adam (lr=3e-4)
- **Loss**: Cross-entropy with gradient clipping (max_norm=1.0)
- **Epochs**: 10
- **Target Loss**: ~1.9

## Progress Log

| Date | Update | Details |
|------|--------|---------|
| Day 1 | Initial architecture | Transformer with RMSNorm + SwiGLU + Flash Attention |

## Requirements

- Python 3.11+
- PyTorch 2.0+ (CUDA recommended)
- tiktoken, flask, numpy, tqdm

## Testing
Run `pytest tests/` to execute the unit test suite.
\n<!-- UI update marker -->\n\n<!-- Hardware specs confirmed -->\n