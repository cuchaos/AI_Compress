import torch
import torch.nn as nn
from transformers import DistilBertModel, MistralForCausalLM

def load_large_model(device="cuda"):
    device = torch.device(device)
    print(f"Loading large model on {device}")
    # large_model 保持 FP16，因為它不參與訓練
    model = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.float16
    ).to(device)
    print(f"Large model memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    return model

def load_small_model(device="cuda"):
    device = torch.device(device)
    print(f"Loading small model on {device}")
    # small_model 使用默認 FP32
    small_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    small_model.classifier = nn.Linear(small_model.config.hidden_size, 256).to(device)  # FP32
    print(f"Small model memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    return small_model

class InstructionDecoder(nn.Module):
    def __init__(self, input_dim=256, target_shape=(1024, 1024)):
        super().__init__()
        self.target_shape = target_shape
        output_dim = target_shape[0] * target_shape[1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # FP32
            nn.ReLU(),
            nn.Linear(256, output_dim),  # FP32
        )
    def forward(self, instruction):
        weights = self.net(instruction)
        return weights.view(*self.target_shape)

def get_instruction(input_ids, small_model):
    outputs = small_model(input_ids)
    cls_output = outputs.last_hidden_state[:, 0, :]
    instruction = small_model.classifier(cls_output)
    return instruction