이 프로젝트는 메인프로젝트와는 조금 떨어진 서브 프로젝트야

[필수규칙]
1. 원본데이터 삭제 혹은 수정 금지. (우리의작업 공간인 "E2B_INT4_MODEL_INFER", "E2B_ORIGINAL_MODEL_INFER"폴더 안의 내용만 삭제, 수정. 그 외 외부폴더 삭제, 수정 금지 단, 참조 나 복사 해서 우리 작업폴더로 가져오는건 가능.)
2. 항상 모델을 굴려볼때 램을 유의하면서 작업 해줘. 스왑램이 32기가라는 용량이 있어도 계속 작업하다 보면 램이 오버 되어서 튕기는 현상을 자주 경험함. 따라서 항상 주의하고. 파이썬 자체로 램 누수가 생길수도 있으니. pynq도 가끔 껐다 켜고 해줘.

파이썬을 실행시키는 방법은 : source /home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/activate
이걸로 pynq_env활성화 그다음에
/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python main.py
하면돼.

'newp' 폴더안의 내용은 gemma3N E4B 모델 그리고 E2B모델을 4500U ram16gb(vram3gb) ubuntu linux, swapRAM: 32gb
환경에서 돌리려는것이고 

핵심은 네가지 폴더야

우선 gemma3N E4B용으로는 
"E4B_ORIGINAL_MODEL_INFER"폴더와 "E4B_INT4_MODEL_INFER"폴더가 있어
두 폴더 모두 절대 수정하지마.

"E4B_ORIGINAL_MODEL_INFER" 폴더 안에 local_gemma_3n 폴더가 있고 
"local_gemma_3n"폴더 그 안에 gemma3N E4B의 safetensor나 json파일들 등등이 존재해, 모델의 구조를 파악하기에 용이하지. 특히 마크다운 파일을 읽어보는거 추천해
다시 "E4B_ORIGINAL_MODEL_INFER" 폴더 안에는 python파일들 .py파일들이 있는데 이 파일들은 gemma3N E4B모델을 추론하는 코드야. 정상 작동하니까 수정하지마.

"E4B_INT4_MODEL_INFER" 폴더 안에는 "E4B_ORIGINAL_MODEL_INFER"폴더의 내용을 보고이를 int4양자화 해서 추론을 해본거야

"E2B_INT4_MODEL_INFER", "E2B_ORIGINAL_MODEL_INFER"이 두 폴더는 우리가 작업할 폴더야.
"E2B_ORIGINAL_MODEL_INFER" 이폴더 는 원본모델 추론이야
"E2B_INT4_MODEL_INFER" 이폴더는 양자화 모델 추론이야

[1]번째로 할거는 "E2B_ORIGINAL_MODEL_INFER" 의 "[Original Model]gemma3NE2B"폴더에는 gemma3N E2B모델이 있어.
그리고 "E4B_ORIGINAL_MODEL_INFER"폴더에는 gemma3N E4B모델 추론용 이지만 파이썬 코드가 있어.
이 코드를 기반으로 "E2B_ORIGINAL_MODEL_INFER"안에 파이썬 파일들을 그대로 만들고 E2B전용으로 수정만 해서 실행되는지 봐줘
gemma3N E 시리즈 모델은 특수한 구조를 가지고 있으니까 E4B모델의 구조를 설명하는 글인 "E4B_ORIGINAL_MODEL_INFER"폴더 안의 "local_gemma_3n" 폴더 안의 마크다운 문서를 읽어보는걸 추천해.

[2]번째로 할거는 "E2B_INT4_MODEL_INFER"폴더야 [1]번 작업이 성공하면 [1]번작업으로 나온 파이썬 파일을 보고 그대로 가져온 다음 "E2B_INT4_MODEL_INFER"안에 붙여넣어 그다음  "E2B_ORIGINAL_MODEL_INFER에 있는 gemma3N E2B 모델을 양자화 해서 "E2B_INT4_MODEL_INFER"안으로 복사해놔 그다음 파이썬 파일을 E2B INT4양자화 전용으로 조금 수정해

[주의]
꼭 출력이 정상적이게 나와야해.
original_model 추론할때보면 정상적으로 forward 토큰 해서 스스로
답변을 매우 논리적이고 문맥에 맞게 한글로 생성을 하니까
양자화 된 모델도 똑같이 의미가 맞고 문맥이 맞고 한글로 나와야해
