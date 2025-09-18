import torch
import tiktoken
from config import ModelConfig
from model import SmallLanguageModel

def generate_text(prompt, max_new_tokens=100):
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SmallLanguageModel(config).to(device)
    
    # Load latest checkpoint if available
    try:
        model.load_state_dict(torch.load(f"slm_epoch_{config.epochs}.pt", map_location=device))
        print("Loaded trained model weights.")
    except Exception as e:
        print("Warning: No trained weights found. Output will be randomized.")
    
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    encoded_prompt = enc.encode(prompt)
    
    print(f"Generating context from prompt: '{prompt}'")
    x = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)
    
    # Autoregressive decoding as defined in the system
    y = model.generate(x, max_new_tokens, temperature=0.8, top_k=50)
    
    decoded_text = enc.decode(y[0].tolist())
    print("\n--- Domain-Specific Generated Text ---")
    print(decoded_text)
    print("--------------------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The system is capable of", help="Initial prompt text")
    parser.add_argument("--tokens", type=int, default=50, help="Number of max new tokens to autocomplete")
    args = parser.parse_args()
    
    generate_text(args.prompt, args.tokens)

# Note: Throughput tracking added for performance monitoring
\n# Kwargs simplified\n