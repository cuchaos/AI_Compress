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

    original_weights = large_model.model.layers[0].self_attn.q_proj.weight.to(device)
    target_shape = decoder.target_shape
    original_weights = original_weights[:target_shape[0], :target_shape[1]]  # FP16

    scaler = GradScaler('cuda')

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 使用 autocast 進行 FP16 計算
        with autocast('cuda', dtype=torch.float16):
            instruction = get_instruction(input_ids, small_model)
            decoded_weights = decoder(instruction)
            loss = criterion(decoded_weights, original_weights)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch}: Loss is NaN or Inf! Stopping training.")
            return small_model, decoder

        # 反向傳播，梯度保持 FP32
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    return small_model, decoder