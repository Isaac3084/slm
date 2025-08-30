from logger import logger
import math
import torch
import torch.optim as optim
from tqdm import tqdm
from config import ModelConfig
from model import SmallLanguageModel
from dataset import get_dataloader

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def train():
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    
    logger.info(f"Config: {config}")
    model = SmallLanguageModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params/1e6:.2f}M")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    dataloader = get_dataloader('train.bin', config.block_size, config.batch_size, stride=128)
    
    max_steps = len(dataloader) * config.epochs
    logger.info(f"Total steps: {max_steps}, Warmup: {config.warmup_steps}")
    
    model.train()
    global_step = 0
    
    logger.info("Starting training...\n")
    for epoch in range(1, config.epochs + 1):
        total_loss = 0
        n_batches = 0
        lr = config.learning_rate
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, targets=y)
            loss = loss / config.grad_accum_steps
            loss.backward()
            
            # Step after accumulation
            if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Update learning rate
                lr = get_lr(global_step, config.warmup_steps, max_steps, 
                           config.learning_rate, config.min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            actual_loss = loss.item() * config.grad_accum_steps
            total_loss += actual_loss
            n_batches += 1
            pbar.set_postfix({'loss': f"{actual_loss:.4f}", 'lr': f"{lr:.2e}"})
        
        avg_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        torch.save(model.state_dict(), f"slm_epoch_{epoch}.pt")
    
    logger.info(f"\nTraining complete. Final loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()
\n# Whitespace normalization\n