# utils.py
import torch.quantization

def quantize_model(model):
    """量化模型以減少記憶體"""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def check_memory():
    """檢查記憶體使用情況"""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("Running on CPU")