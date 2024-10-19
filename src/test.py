import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda" if torch.cuda.is_available() else "cpu")

print(f"模型参数量: {model.num_parameters() / 1e6:.2f}M")
print(f"模型占用显存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
