from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257 # standard GPT-2 byte-pair encoding vocab size
    block_size: int = 512 # sequence length (upgraded from 256)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    learning_rate: float = 3e-4 # Adam optimizer learning rate
    min_lr: float = 3e-5 # cosine scheduler minimum lr
    warmup_steps: int = 100 # warmup before cosine decay
    epochs: int = 10 # 10 epochs targeting ~1.9 final loss
    batch_size: int = 8 # fits 6GB VRAM with block_size=512
    grad_accum_steps: int = 4 # effective batch = 8 * 4 = 32
    device: str = 'cuda' # or 'cpu'
\n# 384 embed dim provides good trade-off for 6 layers\n