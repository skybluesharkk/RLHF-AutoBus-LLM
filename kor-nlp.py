import time

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
sequence_to_classify = input()
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)




# 실행 시간 측정을 위한 시작 시간 기록
start_time = time.time()

# Load the classification pipeline with the specified model
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis",device=0)

# 모델 로드 시간 측정
model_load_time = time.time() - start_time
print(f"Model load time: {model_load_time:.2f} seconds")

# 문장 분류 실행 및 시간 측정
sentence = "나 이 모델 처음 다운받아서 써봐! 쫌 괜찮네 ㅋ"
inference_start = time.time()
result = pipe(sentence)
inference_time = time.time() - inference_start

# 결과 출력
print(f"Inference time: {inference_time:.2f} seconds")
print("Result:", result)
