# main.py
import torch
from models import load_large_model, load_small_model, InstructionDecoder, get_instruction
from train import train
from inference import infer
from transformers import AutoTokenizer

def main():
    device = torch.device("cuda")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

    large_model = load_large_model(device)
    small_model = load_small_model(device)
    target_shape = (4096, 16)  # 更新為 LoRA B 的形狀
    decoder = InstructionDecoder(input_dim=256, target_shape=target_shape, device=device)
    print(f"decoder.net[0].weight.device: {decoder.net[0].weight.device}")
    print(f"decoder.net[0].weight.dtype: {decoder.net[0].weight.dtype}")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    small_model, decoder = train(small_model, decoder, large_model, input_ids, epochs=10, device=device)
    output = infer(small_model, decoder, large_model, input_ids, device=device)
    print("Generated output:", tokenizer.decode(output[0], skip_special_tokens=True))

    torch.save(small_model.state_dict(), "small_model.pt")
    torch.save(decoder.state_dict(), "decoder.pt")

if __name__ == "__main__":
    main()