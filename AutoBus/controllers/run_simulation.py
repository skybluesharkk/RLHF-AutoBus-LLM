import pygame
import numpy as np
from AutoBus.controllers.gpt_controller_v2 import NaturalLanguageController  # GPT 컨트롤러

from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv

# GUI 설정
WIDTH, HEIGHT = 500, 350  # GUI 창 크기
BUTTON_COLOR = (0, 128, 255)  # 파란색
BUTTON_HOVER = (100, 200, 255)  # 밝은 파란색
WHITE = (255, 255, 255)  # 흰색
BLACK = (0, 0, 0)  # 검은색
TEXTBOX_COLOR = (200, 200, 200)  # 입력창 색상

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MetaDrive Controller")
font = pygame.font.Font(None, 32)
input_font = pygame.font.Font(None, 28)  # 입력 폰트

class HybridController:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.nl_controller = NaturalLanguageController()
        self.use_gpt_control = False  # 기본은 PPO 사용
        self.command_queue = []  # GPT 변환된 명령 저장
        self.running = True  # 루프 제어
        self.user_text = ""  # 사용자 입력 저장
        self.typing = False  # 입력 활성화 여부

    def map_command_to_action(self, command, strength):
        """ 
        자연어 명령을 (steering, acceleration) 값으로 변환
        """
        command_mapping = {
            "W": [0, 1.0 * strength],  # 직진 (강도 조절)
            "A": [-0.5 * strength, 0.8],  # 부드러운 좌회전
            "S": [0, -0.5 * strength],  # 감속
            "D": [0.5 * strength, 0.8],  # 부드러운 우회전
        }
        return command_mapping.get(command, [0, 0])  # 기본값: 정지

    def stop_vehicle(self):
        """ 차량을 즉시 정지시키는 함수 """
        self.command_queue = [("S", 0.0)]  # 정지 명령 추가

    def run(self):
        obs, _ = self.env.reset()

        while self.running:
            screen.fill(WHITE)  # 배경색 설정
            mouse_x, mouse_y = pygame.mouse.get_pos()  # 마우스 위치 가져오기
            
            # PPO 버튼
            ppo_rect = pygame.Rect(50, 250, 150, 50)
            pygame.draw.rect(screen, BUTTON_HOVER if ppo_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, ppo_rect)
            ppo_text = font.render("PPO auto", True, WHITE)
            screen.blit(ppo_text, (ppo_rect.x + 30, ppo_rect.y + 15))
            
            # GPT 버튼
            gpt_rect = pygame.Rect(300, 250, 150, 50)
            pygame.draw.rect(screen, BUTTON_HOVER if gpt_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, gpt_rect)
            gpt_text = font.render("use my order", True, WHITE)
            screen.blit(gpt_text, (gpt_rect.x + 20, gpt_rect.y + 15))

            # 입력창 (GPT 모드일 때만 활성화)
            input_box = pygame.Rect(50, 180, 300, 40)
            pygame.draw.rect(screen, TEXTBOX_COLOR, input_box)
            input_surface = input_font.render(self.user_text, True, BLACK)
            screen.blit(input_surface, (input_box.x + 10, input_box.y + 10))

            # 확인(Submit) 버튼
            submit_rect = pygame.Rect(370, 180, 80, 40)
            pygame.draw.rect(screen, BUTTON_HOVER if submit_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, submit_rect)
            submit_text = font.render("확인", True, WHITE)
            screen.blit(submit_text, (submit_rect.x + 20, submit_rect.y + 10))

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if ppo_rect.collidepoint(event.pos):  # PPO 버튼 클릭
                        self.use_gpt_control = False
                        self.command_queue = []  # GPT 명령 초기화
                        print("PPO 자동주행 모드 활성화")

                    elif gpt_rect.collidepoint(event.pos):  # GPT 버튼 클릭
                        self.use_gpt_control = True
                        self.user_text = ""  # 입력창 초기화
                        self.typing = True  # 입력 활성화
                        self.stop_vehicle()  # GPT 버튼 누르면 즉시 정지
                        print("🤖 GPT 수동조작 모드 활성화 (차량 정지)")

                    elif submit_rect.collidepoint(event.pos) and self.typing:  # 확인 버튼 클릭
                        self.typing = False
                        self.nl_controller.get_command(self.user_text)  # 자연어 → 조작 변환
                        self.command_queue = self.nl_controller.command_queue  # 명령 대기열 업데이트
                        print(f"🤖 GPT 변환 결과: {self.command_queue}")

                elif event.type == pygame.KEYDOWN and self.typing:
                    if event.key == pygame.K_BACKSPACE:  # 백스페이스 입력 시 삭제
                        self.user_text = self.user_text[:-1]
                    else:
                        self.user_text += event.unicode  # 입력 추가

            
            if self.use_gpt_control and self.command_queue:
                command, strength = self.command_queue.pop(0)  # (명령어, 강도) 튜플 꺼내기
                action = self.map_command_to_action(command, strength)  
            elif not self.use_gpt_control:
                action, _ = self.model.predict(obs, deterministic=True)  # PPO 자동주행

            # 환경 업데이트
            obs, _, done, _, _ = self.env.step(action)
            self.env.render()

            # PyGame 화면 업데이트
            pygame.display.flip()

            # 주행 종료 조건
            if done:
                obs, _ = self.env.reset()

        self.env.close()
        pygame.quit()  # PyGame 종료

# 실행 코드
if __name__ == "__main__":
    env = MetaDriveEnv(dict(use_render=True))
    model = PPO.load("ppo_metadrive_test_mycode.zip")  # 학습된 모델 로드
    controller = HybridController(env, model)
    controller.run()
