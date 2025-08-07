import os
import urllib.request
import tiktoken
import numpy as np

urls = [
    # Original sources (~642K tokens)
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    # Additional sources to reach 1.8M+ tokens
    "https://www.gutenberg.org/files/2600/2600-0.txt",  # War and Peace (~580K tokens)
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick (~215K tokens)
    "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations (~185K tokens)
    "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula (~165K tokens)
    "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities (~140K tokens)
    "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre (~185K tokens)
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Adventures of Sherlock Holmes (~115K tokens)
    "https://www.gutenberg.org/files/768/768-0.txt",    # Wuthering Heights (~115K tokens)
    "https://www.gutenberg.org/files/174/174-0.txt",    # The Picture of Dorian Gray (~80K tokens)
]

def build_nlp_pipeline():
    # Advanced NLP: High performance byte-pair encoding
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    
    print("Downloading and tokenizing high-density NLP datasets...")
    for url in urls:
        print(f"Fetching {url.split('/')[-1]}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            text = response.read().decode('utf-8', errors='ignore')
            
            # Using encode_ordinary which avoids regex matching safety limits for massive text
            tokens = enc.encode_ordinary(text)
            all_tokens.extend(tokens)
            print(f" -> Merged {len(tokens):,} tokens.")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    # Memory Map array construction for RAM-efficient streaming (Pro tier NLP)
    train_tokens = all_tokens[:int(len(all_tokens)*0.95)]
    val_tokens = all_tokens[int(len(all_tokens)*0.95):]
    
    # uint16 supports up to 65,535 vocab size, our vocab is 50257 so this perfectly fits!
    train_tokens_np = np.array(train_tokens, dtype=np.uint16)
    val_tokens_np = np.array(val_tokens, dtype=np.uint16)
    
    train_tokens_np.tofile('train.bin')
    val_tokens_np.tofile('val.bin')
    
    print(f"\nNLP Tokenization Pipeline Complete!")
    print(f"Total Combined Corpus Size: {len(all_tokens):,} tokens mapped into high-speed binary streams.")
    
    if len(all_tokens) >= 1_800_000:
        print(f"✅ PASSED: Corpus exceeds 1.8M token threshold ({len(all_tokens):,} tokens)")
    else:
        print(f"⚠️ WARNING: Corpus is {len(all_tokens):,} tokens — below 1.8M target. Need {1_800_000 - len(all_tokens):,} more tokens.")

if __name__ == '__main__':
    build_nlp_pipeline()
