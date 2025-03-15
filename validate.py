# validate.py
import torch
import time
import logging
from models import load_large_model, load_small_model, InstructionDecoder, get_instruction
from inference import infer
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log(message):
    print(message)
    logging.info(message)

def validate(small_model, decoder, large_model, val_input_ids, device):
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            instruction = get_instruction(val_input_ids, small_model)
            decoded_weights = decoder(instruction).to(dtype=torch.float16)  # 形狀: [1, 4096, 16]
            
            q_proj = large_model._orig_mod.base_model.model.model.layers[0].self_attn.q_proj
            if 'default' in q_proj.lora_B:
                original_weights = q_proj.lora_B['default'].weight.to(device, dtype=torch.float16)
            else:
                raise ValueError("未找到預期的 LoRA 'default' 鍵")
            
            loss = torch.nn.MSELoss(reduction='mean')(decoded_weights, original_weights.unsqueeze(0))
    return loss.item()

# 以下函數保持不變
def test_memory_usage(small_model, decoder, large_model, tokenizer, device):
    input_lengths = [10, 50, 500]
    batch_sizes = [1, 2, 4]
    for length in input_lengths:
        for batch in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            input_text = "The future of AI is " + "word " * (length - 5)
            inputs = tokenizer([input_text] * batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            output = infer(small_model, decoder, large_model, input_ids, attention_mask, device)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            log(f"Length: {length}, Batch: {batch}, Peak Memory: {peak_memory:.2f} GB")

def test_generation_quality(small_model, decoder, large_model, tokenizer, device):
    prompts = ["What is AI?", "AI is transforming the world because", "The future of AI is"]
    smooth = SmoothingFunction().method1
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output = infer(small_model, decoder, large_model, input_ids, attention_mask, device)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        baseline_output = large_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=200)
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        bleu_score = sentence_bleu([baseline_text.split()], generated_text.split(), smoothing_function=smooth)
        log(f"Prompt: {prompt}\nGenerated: {generated_text}\nBaseline: {baseline_text}\nBLEU: {bleu_score:.4f}\n")

def test_generalization(small_model, decoder, large_model, tokenizer, device):
    val_prompts = ["Climate change is a pressing issue because", "The history of technology shows", "In the next decade, space exploration will"]
    total_loss = 0
    for prompt in val_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        loss = validate(small_model, decoder, large_model, input_ids, device)
        total_loss += loss
        log(f"Prompt: {prompt}, Loss: {loss:.6f}")
    avg_loss = total_loss / len(val_prompts)
    log(f"Average Validation Loss: {avg_loss:.6f}")

def test_stability(small_model, decoder, large_model, tokenizer, device, runs=5):
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    losses = []
    for i in range(runs):
        torch.cuda.empty_cache()
        loss = validate(small_model, decoder, large_model, input_ids, device)
        output = infer(small_model, decoder, large_model, input_ids, attention_mask, device)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        losses.append(loss)
        log(f"Run {i+1}: Loss: {loss:.6f}, Generated: {generated_text}")
    avg_loss = sum(losses) / runs
    std_loss = (sum((l - avg_loss) ** 2 for l in losses) / runs) ** 0.5
    log(f"Average Loss: {avg_loss:.6f}, Std Dev: {std_loss:.6f}")

def test_efficiency(small_model, decoder, large_model, tokenizer, device):
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    start_time = time.time()
    output = infer(small_model, decoder, large_model, input_ids, attention_mask, device)
    end_time = time.time()
    log(f"Inference Time: {end_time - start_time:.2f} seconds")

def main():
    device = torch.device("cuda")
    log(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()

    large_model = load_large_model(device)
    small_model = load_small_model(device)
    target_shape = (4096, 16)  # 更新為 LoRA B 的形狀
    decoder = InstructionDecoder(input_dim=256, target_shape=target_shape, device=device)

    small_model.load_state_dict(torch.load("small_model.pt", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load("decoder.pt", map_location=device, weights_only=True))
    small_model.eval()
    decoder.eval()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    large_model.generation_config.pad_token_id = tokenizer.pad_token_id

    log("\n=== Memory Usage Test ===")
    test_memory_usage(small_model, decoder, large_model, tokenizer, device)
    
    log("\n=== Generation Quality Test ===")
    test_generation_quality(small_model, decoder, large_model, tokenizer, device)
    
    log("\n=== Generalization Test ===")
    test_generalization(small_model, decoder, large_model, tokenizer, device)
    
    log("\n=== Stability Test ===")
    test_stability(small_model, decoder, large_model, tokenizer, device)
    
    log("\n=== Efficiency Test ===")
    test_efficiency(small_model, decoder, large_model, tokenizer, device)

if __name__ == "__main__":
    main()