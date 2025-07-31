from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import torch
import warnings
import os

# From gemma repo

# Set environment variables to reduce warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Disable TensorFloat32 warning by setting high precision
torch.set_float32_matmul_precision('high')

# Disable PyTorch inductor compilation to avoid CUDA graph warnings
torch._dynamo.config.disable = True

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"  # Use eager attention to avoid compilation issues
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Clear problematic generation config parameters
model.generation_config.top_p = None
model.generation_config.top_k = None
model.generation_config.temperature = None

messages = [
    {"role": "user", "content": "Describe what sentences are."}
]

text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(
        **inputs, 
        max_new_tokens=100, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)