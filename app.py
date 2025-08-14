import os
import torch
import tiktoken
from flask import Flask, request, jsonify, render_template
from config import ModelConfig
from model import SmallLanguageModel

app = Flask(__name__)

# Initialize model
config = ModelConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallLanguageModel(config).to(device)
enc = tiktoken.get_encoding("gpt2")

checkpoint_path = f"slm_epoch_{config.epochs}.pt"
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Filter out keys with shape mismatches (e.g. positional embeddings after context window upgrade)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"Loaded weights (partial) — skipped {skipped} due to shape mismatch. Retrain needed.")
    else:
        print(f"Loaded weights from {checkpoint_path}")
else:
    print("Warning: No trained weights found. Using random weights.")

model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = min(data.get('max_tokens', 100), 500)
    temperature = data.get('temperature', 0.8)
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
        
    try:
        encoded = enc.encode(prompt)
        x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            y = model.generate(x, max_tokens, temperature=temperature, top_k=50)
            
        return jsonify({'generated_text': enc.decode(y[0].tolist())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """Sentiment classification — placeholder until classifier head is trained."""
    return jsonify({'error': 'Classifier not trained yet. Coming soon!'}), 501

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat mode — placeholder until instruction tuning is done."""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # For now, use the base model with a chat-like prompt format
    chat_prompt = f"Human: {message}\nAssistant:"
    
    try:
        encoded = enc.encode(chat_prompt)
        x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=150, temperature=0.7, top_k=50)
        
        full_text = enc.decode(y[0].tolist())
        # Extract just the assistant reply
        reply = full_text.split("Assistant:")[-1].strip()
        # Cut at next "Human:" if present
        if "Human:" in reply:
            reply = reply.split("Human:")[0].strip()
        
        return jsonify({'reply': reply if reply else 'I need more training to answer that properly.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
