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
        사용자의 자연어 input -> GPT API -> 차량 조작 명령(action)과 보상(reward)을 변환
        반환하는 JSON의 구조:
        {
            "action": { "command": "<W, A, S, D 중 하나>", "strength": <0.0 ~ 1.0> },
            "reward": <정수>
        }
        예를 들어, 입력 "별로야"에 대해 아래와 같이 출력:
        {
            "action": { "command": "", "strength": 0 },
            "reward": -2
        }
        """

        prompt = f"""
        사용자의 자연어 명령을 차량 조작 명령(action)과 보상(reward)으로 변환하세요.
        반환하는 JSON의 구조는 반드시 아래와 같아야 합니다:

        {{
            "action": {{ "command": "<W, A, S, D 중 하나>", "strength": <0.0 ~ 1.0> }},
            "reward": <정수>
        }}

        예시 매핑:

            - "앞으로 가" → {{ "action": {{ "command": "W", "strength": 0.8 }}, "reward": 0 }}
            - "잘했어 정말 최고야" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": 2 }}
            - "앞으로 가, 잘했어" → {{ "action": {{ "command": "W", "strength": 0.8 }}, "reward": 2 }}
            - "별로야" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": -2 }}
            - "그게 뭐야 실망인데" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": -2 }}
            - "뭐하지?" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": 0 }}
            - "오른쪽으로 돌아, 별로야" → {{ "action": {{ "command": "D", "strength": 1.0 }}, "reward": -2 }}
            - "앞으로 가, 그거 별로야" → {{ "action": {{ "command": "W", "strength": 0.8 }}, "reward": -2 }}
            - "빨리 가" → {{ "action": {{ "command": "W", "strength": 1.0 }}, "reward": 0 }}
            - "천천히 가, 잘해" → {{ "action": {{ "command": "W", "strength": 0.3 }}, "reward": 2 }}
            - "멈춰, 이거 너무 별로야" → {{ "action": {{ "command": "S", "strength": 0.0 }}, "reward": -2 }}
            - "조용히 해" → {{ "action": {{ "command": "S", "strength": 0.0 }}, "reward": 0 }}
            - "좋은 선택이야" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": 2 }}
            - "그건 틀렸어" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": -2 }}
            - "어떻게 해야 할지 모르겠어" → {{ "action": {{ "command": "", "strength": 0 }}, "reward": 0 }}
            - "오른쪽으로 가, 최고야" → {{ "action": {{ "command": "D", "strength": 0.8 }}, "reward": 2 }}
            - "멈춰, 너무 빨라서 위험해" → {{ "action": {{ "command": "S", "strength": 0.0 }}, "reward": -2 }}
            - "정말 대단해, 계속 가" → {{ "action": {{ "command": "W", "strength": 0.8 }}, "reward": 2 }}
            - "천천히 왼쪽으로 가" → {{ "action": {{ "command": "A", "strength": 0.3 }}, "reward": 0 }}
            - "오른쪽으로 급격히 회전" → {{ "action": {{ "command": "D", "strength": 1.0 }}, "reward": 0 }}
            - "전진해" → {{ "action": {{ "command": "W", "strength": 0.8 }}, "reward": 0 }}
            - "정지" → {{ "action": {{ "command": "S", "strength": 0.0 }}, "reward": 0 }}

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
        '''
        if gpt_output.startswith("```"):
            gpt_output = "\n".join(gpt_output.splitlines()[1:-1]).strip()
        '''
        try:
            parsed_output = json.loads(gpt_output)  
            action_dict = parsed_output.get("action", {"command": "S", "strength": 0})
            if action_dict.get("command") is None:
                action_dict["command"] = "S"
            command = action_dict.get("command", "S")
            strength = action_dict.get("strength", 0)
            reward = parsed_output.get("reward", 0)
            output_json = {"action": {"command": command, "strength": strength}, "reward": reward}
            self.command_queue.append(output_json)
            return output_json
        except json.JSONDecodeError:
            print(f"JSON 변환 실패: {gpt_output}")
            fallback = {"action": {"command": "S", "strength": 0}, "reward": 0}
            self.command_queue.append(fallback)
            return fallback

    def process_input(self):
        """ 명령 큐에서 하나씩 꺼내기 """
        if self.command_queue:
            return self.command_queue.pop(0)
        return {"action": {"command": "S", "strength": 0}, "reward": 0}
