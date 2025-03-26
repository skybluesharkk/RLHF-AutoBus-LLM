import os
import json
from dotenv import load_dotenv
from transformers import pipeline

# 환경변수 로드
load_dotenv()

# 행동 분류 파이프라인 (Zero-shot classification)
action_classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# 감성 분석 파이프라인 (Text classification)
reward_analyzer = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    device=0  # GPU 사용, 없으면 제거
)

class NaturalLanguageController:
    def __init__(self):
        self.command_queue = []  

    def get_command(self, user_input):
        """
        사용자의 자연어 피드백을 받아 Action(차량 조작)과 Reward(보상)를 분리하여 반환.
        각 모델의 결과(라벨과 신뢰도)를 출력.
        """
        # Action 분류 (Zero-shot classification)
        candidate_actions = ["Move Forward", "Turn Left", "Turn Right", "Stop", "Increase Speed", "Decrease Speed"]
        action_output = action_classifier(user_input, candidate_actions, multi_label=False)
        
        # action_output에는 "labels"와 "scores"가 포함.
        action_label = action_output["labels"][0]  # 가장 높은 확률의 라벨
        action_confidence = action_output["scores"][0]  # 해당 라벨의 신뢰도
        
        # 모델 응답 출력
        print("Action Classifier:")
        print("  후보 라벨:", action_output["labels"])
        print("  신뢰도:", action_output["scores"])

        # Reward 분석 (Sentiment Analysis)
        reward_output = reward_analyzer(user_input)
        reward_score = reward_output[0]["score"]
        reward_label = reward_output[0]["label"]

        # 모델 응답 출력
        print("Reward Analyzer:")
        print("  라벨:", reward_label)
        print("  점수:", reward_score)

        # 감성 분석 결과를 강화학습 보상값으로 변환
        if reward_label.lower() == "positive" or "very positive":
            reward = round(reward_score, 2)  # 0.0 ~ 1.0
        elif reward_label.lower() == "negative" or "very negative":
            reward = round(-reward_score, 2)  # -1.0 ~ 0.0
        else:
            reward = 0.0

        # 최종 결과 출력 (보상 포함)
        print("최종 보상:", reward)

        # 큐에 명령 추가 및 결과 반환
        self.command_queue.append((action_label, action_confidence, reward))
        return action_label, action_confidence, reward

    def process_input(self):
        """ 명령 큐에서 하나씩 꺼내기 """
        if self.command_queue:
            return self.command_queue.pop(0)
        return ("Stop", 0.0, 0.0)  # 기본 정지 + 보상 없음

if __name__ == "__main__":
    controller = NaturalLanguageController()
    user_input = input("자연어 피드백 입력 : ")

    action, confidence, reward = controller.get_command(user_input)
    print(f"\n분석 결과:\n  Action: {action}\n  Confidence: {confidence:.2f}\n  Reward: {reward}")
