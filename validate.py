# validate.py
import torch
from models import load_large_model, load_small_model, InstructionDecoder, get_instruction
from inference import infer
from transformers import AutoTokenizer

def validate(small_model, decoder, large_model, val_input_ids, device):
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            instruction = get_instruction(val_input_ids, small_model)
            decoded_weights = decoder(instruction)
            original_weights = large_model.model.layers[0].self_attn.q_proj.weight[:decoder.target_shape[0], :decoder.target_shape[1]]
            loss = torch.nn.MSELoss()(decoded_weights, original_weights)
    return loss.item()

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()

    # 加載模型
    large_model = load_large_model(device)
    small_model = load_small_model(device)
    target_shape = (1024, 1024)
    decoder = InstructionDecoder(input_dim=256, target_shape=target_shape).to(device)

    # 加載訓練好的權重
    small_model.load_state_dict(torch.load("small_model.pt", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load("decoder.pt", map_location=device, weights_only=True))
    small_model.eval()
    decoder.eval()

    # 準備真實輸入
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token  # 設置 pad_token
    large_model.config.pad_token_id = tokenizer.pad_token_id  # 同步到模型配置
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 驗證損失
    val_loss = validate(small_model, decoder, large_model, input_ids, device)
    print(f"Validation Loss: {val_loss}")

    # 推理並生成
    output = infer(small_model, decoder, large_model, input_ids, attention_mask=attention_mask, device=device)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

    # 基準生成
    baseline_output = large_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,  # 增加生成長度
        repetition_penalty=1.2,
        do_sample=True,
        temperature=0.9       # 增加隨機性
    )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print("Baseline text:", baseline_text)

    # 記憶體峰值
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()