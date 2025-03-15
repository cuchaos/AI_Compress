# models.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model

def load_large_model(device):
    print(f"Loading large model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        load_in_4bit=True,  # 4-bit 量化
        device_map="auto",
        torch_dtype=torch.float16
    )
    # 應用 LoRA
    lora_config = LoraConfig(
        r=16,  # 秩
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # 目標注意力層
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    # 啟用 FlashAttention（若硬體支援）
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        model.config.use_flash_attention_2 = True
    # 編譯模型
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    memory = torch.cuda.memory_allocated(device) / 1024**3
    print(f"Large model memory: {memory:.2f} GB")
    return model

def load_small_model(device):
    print(f"Loading small model on {device}")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    model.classifier = nn.Linear(model.config.dim, 256).to(device)
    memory = torch.cuda.memory_allocated(device) / 1024**3
    print(f"Small model memory: {memory:.2f} GB")
    return model

class InstructionDecoder(nn.Module):
    def __init__(self, input_dim=256, target_shape=(1024, 1024)):
        super().__init__()
        self.target_shape = target_shape
        output_dim = target_shape[0] * target_shape[1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, instruction):
        weights = self.net(instruction)
        batch_size = instruction.size(0)
        return weights.view(batch_size, *self.target_shape)

def get_instruction(input_ids, small_model):
    outputs = small_model(input_ids)
    cls_output = outputs.last_hidden_state[:, 0, :]
    instruction = small_model.classifier(cls_output)
    return instruction