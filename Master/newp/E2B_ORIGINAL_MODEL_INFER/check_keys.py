from safetensors.torch import load_file
import glob
import os

model_dir = "E2B_ORIGINAL_MODEL_INFER/[Original Model]gemma3NE2B"
st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

for f in st_files:
    print(f"Checking {f}...")
    tensors = load_file(f)
    for k in tensors.keys():
        if "tokens" in k:
            print(f"  FOUND: {k} (shape: {tensors[k].shape})")
