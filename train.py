# train.py
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.amp import autocast, GradScaler
from models import get_instruction

def train(small_model, decoder, large_model, input_ids, epochs=10, device="cuda"):
    device = torch.device(device)
    print(f"Training on {device}")
    
    optimizer = Adam(list(small_model.parameters()) + list(decoder.parameters()), lr=1e-5)
    criterion = MSELoss()

    # 獲取 q_proj 層的 LoRA B 權重
    q_proj = large_model._orig_mod.base_model.model.model.layers[0].self_attn.q_proj
    if 'default' in q_proj.lora_B:
        original_weights = q_proj.lora_B['default'].weight.to(device, dtype=torch.float16)
    else:
        raise ValueError("未找到預期的 LoRA 'default' 鍵")

    scaler = GradScaler('cuda')

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        with autocast('cuda', dtype=torch.float16):
            instruction = get_instruction(input_ids, small_model)
            decoded_weights = decoder(instruction)  # 形狀: [1, 4096, 16]
            decoded_weights = decoded_weights.to(dtype=torch.float16)
            # print(f"decoded_weights shape: {decoded_weights.shape}")  # 診斷輸出
            # print(f"original_weights shape: {original_weights.shape}")  # 診斷輸出
            loss = criterion(decoded_weights, original_weights.unsqueeze(0))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch}: Loss is NaN or Inf! Stopping training.")
            return small_model, decoder

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    return small_model, decoder