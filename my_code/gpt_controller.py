from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# .env에서 API Key 가져오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not exist.")

client = OpenAI(api_key=api_key)  

class NaturalLanguageController:
    def __init__(self):
        self.command_queue = []  

    def get_command(self, user_input):
        """
        사용자의 자연어 input -> GPT API -> 차량 조작 명령(W, A, S, D)으로 변환
        JSON 형태로 반환받아 처리
        """
        prompt = f"""
        사용자의 자연어 명령을 차량 조작 명령으로 변환하세요.
        명령은 JSON 형식으로 반환해야 합니다.

        예제:
        - "천천히 왼쪽으로 가" → {{ "command": "A", "strength": 0.3 }}
        - "오른쪽으로 급격히 회전" → {{ "command": "D", "strength": 1.0 }}
        - "전진해" → {{ "command": "W", "strength": 0.8 }}
        - "정지" → {{ "command": "S", "strength": 0.0 }}

        아래 입력을 변환하세요:
        입력: "{user_input}"
        출력:
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 자동차 조작 전문가입니다. 사용자의 자연어 입력을 JSON 형태로 변환하세요."},
                {"role": "user", "content": prompt}
            ]
        )

        gpt_output = response.choices[0].message.content.strip()

        try:
            parsed_output = json.loads(gpt_output)  # JSON 파싱
            command = parsed_output.get("command", "S")  # 기본값: 정지
            strength = parsed_output.get("strength", 0.0)  # 기본값: 0
            self.command_queue.append((command, strength))  # 튜플 형태로 저장
        except json.JSONDecodeError:
            print(f"JSON 변환 실패: {gpt_output}")
            self.command_queue.append(("S", 0.0))  # 오류 발생 시 정지

    def process_input(self):
        """ 명령 큐에서 하나씩 꺼내기 """
        if self.command_queue:
            return self.command_queue.pop(0)  
        return ("S", 0.0)  # 기본 정지
