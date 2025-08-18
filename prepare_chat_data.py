import json
import urllib.request
import os
import tiktoken
import numpy as np

def prepare_chat_data():
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    filename = "alpaca_data.json"
    
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} examples.")
    
    # We'll use a subset for quick fine-tuning (e.g., first 10,000 examples)
    subset = data[:10000]
    
    # Format as conversation
    # "Human: ...\nAssistant: ..."
    formatted_text = ""
    for item in subset:
        instruction = item['instruction']
        if item.get('input'):
            instruction += f"\n{item['input']}"
        output = item['output']
        
        # Using a simple prompt format
        formatted_text += f"Human: {instruction}\nAssistant: {output}\n\n"

    print(f"Formatted text length: {len(formatted_text)} characters.")

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(formatted_text)
    print(f"Total tokens: {len(tokens)}")

    # Split 95/5 train/val
    split_idx = int(len(tokens) * 0.95)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens: {len(val_tokens)}")

    # Save to binary
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    
    train_arr.tofile('train_chat.bin')
    val_arr.tofile('val_chat.bin')
    print("Saved to train_chat.bin and val_chat.bin.")

if __name__ == '__main__':
    prepare_chat_data()
