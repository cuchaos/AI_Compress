# inference.py
import torch
from torch.amp import autocast
from models import get_instruction

def infer(small_model, decoder, large_model, input_ids, attention_mask=None, device="cuda"):
    device = torch.device(device)
    print(f"Inferring on {device}")
    
    with autocast('cuda', dtype=torch.float16):
        instruction = get_instruction(input_ids, small_model)
        new_weights = decoder(instruction)  # (batch_size, 1024, 1024)
        # 只使用第一個批次的權重更新 large_model（假設批量推理僅用於測試）
        large_model.model.layers[0].self_attn.q_proj.weight.data[:1024, :1024] = new_weights[0]
    
    output = large_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,
        repetition_penalty=1.2,
        do_sample=True,
        temperature=0.9
    )
    print(f"GPU Memory after inference: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    return output