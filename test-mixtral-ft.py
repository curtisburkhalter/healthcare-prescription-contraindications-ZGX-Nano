from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, "./mixtral_medical_lora")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Test it
prompt = "[INST] What are the drug interactions between warfarin and aspirin? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")

# FIX: Move inputs to the same device as the model
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
