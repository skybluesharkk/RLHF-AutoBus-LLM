import os
import json
from dotenv import load_dotenv


action_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")  # 행동 분류
reward_analyzer = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", device=0)  # 감성 분석

class NaturalLanguageController:
    def __init__(self):
        self.command_queue = []  

    def process_input(self):
        """ 명령 큐에서 하나씩 꺼내기 """
        if self.command_queue:
            return self.command_queue.pop(0)  
        return ("S", 0.0)  # 기본 정지
    
    def get_command(self, user_input):
        """
        사용자의 자연어 피드백을 받아 Action(차량 조작)과 Reward(보상)를 분리하여 반환.
        """
        # 1️⃣ Action 분류 (Zero-shot classification)
        candidate_actions = ["Move Forward", "Turn Left", "Turn Right", "Stop", "Increase Speed", "Decrease Speed"]
        action_output = action_classifier(user_input, candidate_actions, multi_label=False)
        action = action_output["labels"][0]  # 가장 확률 높은 행동 선택
        action_confidence = action_output["scores"][0]  # 신뢰도

        # 2️⃣ Reward 분석 (Sentiment Analysis)
        reward_output = reward_analyzer(user_input)
        reward_score = reward_output[0]["score"]
        reward_label = reward_output[0]["label"]

        # 감성 분석 결과를 강화학습 보상값으로 변환
        if reward_label == "positive":
            reward = round(reward_score, 2)  # 0.0 ~ 1.0
        elif reward_label == "negative":
            reward = round(-reward_score, 2)  # -1.0 ~ 0.0 (부정적이면 감점)
        else:
            reward = 0.0  # 중립적인 피드백은 보상 없음

        # 큐에 명령 추가
        self.command_queue.append((action, action_confidence, reward))
        return action, action_confidence, reward
    def process_input(self):
        """ 명령 큐에서 하나씩 꺼내기 """
        if self.command_queue:
            return self.command_queue.pop(0)  
        return ("Stop", 0.0, 0.0)  # 기본 정지 + 보상 없음