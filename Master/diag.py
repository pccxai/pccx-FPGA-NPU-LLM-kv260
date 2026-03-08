import glob
from safetensors.torch import load_file
import SYS_CONFIG

st_files = sorted(glob.glob(f"{SYS_CONFIG.MODEL_DIR}/*.safetensors"))
if not st_files:
    print("파일이 없습니다!")
else:
    print(f"첫 번째 파일 분석: {st_files[0]}")
    tensors = load_file(st_files[0], device="cpu")
    print("\n[저장된 키 이름 TOP 20개 확인]")
    for k in list(tensors.keys())[:20]:
        print(f" - {k}")