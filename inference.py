# inference.py
import torch
from torch.amp import autocast
from models import get_instruction

def infer(small_model, decoder, large_model, input_ids, attention_mask=None, device="cuda"):
    device = torch.device(device)
    # print(f"Inferring on {device}")
    
    with autocast('cuda', dtype=torch.float16):
        instruction = get_instruction(input_ids, small_model)
        new_weights = decoder(instruction).to(dtype=torch.float16)  # 形狀: [1, 4096, 16]
        # print(f"new_weights shape: {new_weights.shape}")

        # 獲取 q_proj 層
        q_proj = large_model._orig_mod.base_model.model.model.layers[0].self_attn.q_proj

        # 更新 LoRA 權重
        if hasattr(q_proj, 'lora_B'):
            with torch.no_grad():
                if 'default' in q_proj.lora_B:
                    lora_b_weight = q_proj.lora_B['default'].weight
                    # print(f"LoRA B weight shape: {lora_b_weight.shape}")
                    lora_b_weight.data.copy_(new_weights[0])
                else:
                    raise ValueError("未找到預期的 LoRA 'default' 鍵，請檢查 LoRA 配置")
        else:
            raise ValueError("模型未應用 LoRA，無法更新權重")
    
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