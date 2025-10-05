import torch
import torch.onnx
from config import ModelConfig
from model import SmallLanguageModel
import os

def export_to_onnx():
    print("Preparing Mobile-Optimized SLM Export...")
    
    # 1. Load config and model
    config = ModelConfig()
    device = torch.device('cpu') # Exporting for CPU/Edge
    config.device = 'cpu'
    
    model = SmallLanguageModel(config).to(device)
    
    # Check for latest chat-tuned model or fallback
    checkpoint = "slm_chat_epoch_3.pt"
    if not os.path.exists(checkpoint):
        checkpoint = "slm_epoch_10.pt"
        
    print(f"Loading weights from {checkpoint}...")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=False)
    model.eval()

    # Create an ONNX wrapper to handle the tuple return
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model
        def forward(self, idx):
            logits, _ = self.model(idx)
            return logits
            
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()
    
    # 2. Export base model to ONNX
    print("Exporting float32 model to ONNX format...")
    dummy_input = torch.zeros(1, 1, dtype=torch.long)
    onnx_filename = "slm_mobile_fp32.onnx"
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    # 3. Quantize using ONNX Runtime
    print("Applying INT8 dynamic quantization for edge devices...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    onnx_quantized_filename = "slm_mobile_int8.onnx"
    quantize_dynamic(
        onnx_filename,
        onnx_quantized_filename,
        weight_type=QuantType.QUInt8
    )
    
    original_size = os.path.getsize(checkpoint) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_quantized_filename) / (1024 * 1024)
    
    print(f"\\n✅ Mobile Edge Export Complete!")
    print(f"Original PyTorch Size: {original_size:.2f} MB")
    print(f"Optimized Mobile Size: {onnx_size:.2f} MB")
    print(f"Reduction:             {100 - (onnx_size/original_size * 100):.1f}%")
    print(f"Saved to: {onnx_quantized_filename}")
    
    # Clean up temp fp32 file
    if os.path.exists(onnx_filename):
        os.remove(onnx_filename)

if __name__ == '__main__':
    export_to_onnx()
