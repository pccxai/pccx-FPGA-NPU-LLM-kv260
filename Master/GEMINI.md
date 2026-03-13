# Master Directory Guide (Python Software & Controller)

## 1. Execution Environment
* **Virtual Environment:** pynq_env
* **Activation Command:** `source /home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/activate`
* **Execution Command:** `/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python`

## 2. Current Status & Goals
* **[현재목표]:** 공식 모델인 `google/gemma-3n-E2B-it` 모델을 로컬 환경에서 완벽한 채팅 스트리밍 형태로 구동 성공시키기. 
'[Original Model]gemma3NE2B' 그리고 '[Original Model]gemma3NE2B_INT4_Q' 가 공식모델이야
이 폴더들 안에 config.json부터 tokenizer.json 등등 모델의 구조를 파악하기 위한 파일들이 존재해.
단순한 출력 수준이 아니라 gpt, gemini같은 실제 상용화 모델처럼 대화가 자연스럽게 이어져야해
자료가 필요하면 공식문서나 인터넷검색등을 적극적으로 활용해서 검증하고 확인해봐.꼭  debugging.txt에 메모하고!

* **[보류]:** `gemma3N E2B abilterated` 모델을 INT4로 양자화(Quantization)하여 추론하는 파이썬 코드 완성 및 검증.
'[abliterated Model]gemma3NE2B_INT4_Q' 와 '[abliterated Model]gemma3NE2B' 가 보류된 모델이야

## 3. Strict Rules: debugging.txt
* 코드를 수정하거나 에러가 발생할 때 혹은 모델관련 특이사항, 모델구조, 등등 지식을 얻을때 마다 반드시 `debugging.txt` 파일에 기록(메모)할 것.
* **기록 양식:** [시도한 내용] - [발생한 문제/에러 로그] - [해결 방법 및 결과]
* 동일한 실수를 반복하지 않도록 코딩 전 `debugging.txt`를 메모장처럼 확인하고 컨텍스트를 유지할 것.
