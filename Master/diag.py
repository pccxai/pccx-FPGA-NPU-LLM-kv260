# 별도 스크립트로 실행
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"/home/hwkim/Desktop/github/TinyNPU-RTL/Master/gemma3NE2B/"  # 원본 (양자화 전) 폴더
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

prompt = "<start_of_turn>user\n안녕 하세요<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0, -1, :]
top5 = torch.argsort(logits, descending=True)[:5]
print("=== HuggingFace 정답 ===")
for tid in top5.tolist():
    print(f"  토큰 {tid}: {repr(tokenizer.decode([tid]))} 점수: {logits[tid]:.3f}")