import multiprocessing
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from metadrive.envs import MetaDriveEnv
from controllers.llm.gpt.gpt_controller_v2 import NaturalLanguageController

USE_GPT_CONTROL = False
LLM_FEEDBACK = None  # 예: {"action": {"command": "W", "strength": 0.8}, "reward": 2}

def compute_combined_reward(env_reward, llm_reward, alpha=1.0, beta=1.0):
    """ 원래 환경 reward와 LLM 피드백 reward를 가중합 """
    return alpha * env_reward + beta * llm_reward

# 환경 래퍼: MetaDriveEnv에 LLM 피드백을 반영
class LLMFeedbackControlWrapper(MetaDriveEnv):
    def __init__(self, config, shared_feedback):
        super().__init__(config)
        self.shared_feedback = shared_feedback

    def map_command_to_action(self, command, strength):
        mapping = {
            "W": [0, 1.0 * strength],
            "A": [-0.5 * strength, 0.8],
            "S": [0, -0.5 * strength],
            "D": [0.5 * strength, 0.8],
        }
        return mapping.get(command, [0, 0])

    def step(self, action):
        global USE_GPT_CONTROL, LLM_FEEDBACK
        if self.shared_feedback.get("use_gpt_control", False) and "feedback" in self.shared_feedback:
            feedback = self.shared_feedback["feedback"]
            gpt_action = feedback.get("action", {})
            command = gpt_action.get("command", "")
            strength = gpt_action.get("strength", 0)
            action = self.map_command_to_action(command, strength)
            llm_reward = feedback.get("reward", 0)
        else:
            llm_reward = 0
        obs, reward, terminated, truncated, info = super().step(action) #상위 step()에서 reward를 먼저 받아옴
        combined_reward = reward + 0.1 * llm_reward
        print(f"[Env] Action: {action}, Original reward: {reward}, LLM reward: {llm_reward}, Combined reward: {combined_reward}")
        return obs, combined_reward, terminated, truncated, info

# GUI 프로세스: 별도의 프로세스에서 실행
def gui_process(shared_feedback):
    import pygame
    pygame.init()
    WIDTH, HEIGHT = 500, 350
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Training Controller")

    korean_font_path = "AutoBus/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"

    font = pygame.font.Font(korean_font_path, 24)
    input_font = pygame.font.Font(korean_font_path, 20)

    BUTTON_COLOR = (0, 128, 255)
    BUTTON_HOVER = (100, 200, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    TEXTBOX_COLOR = (200, 200, 200)

    clock = pygame.time.Clock()
    running = True
    user_text = ""
    typing = False


    ppo_rect = pygame.Rect(50, 250, 120, 40)
    gpt_rect = pygame.Rect(300, 250, 120, 40)
    input_box = pygame.Rect(50, 180, 250, 35)
    submit_rect = pygame.Rect(320, 180, 80, 35)

    nl_controller = NaturalLanguageController()

    while running:
        screen.fill(WHITE)
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # PPO 버튼
        pygame.draw.rect(screen, BUTTON_HOVER if ppo_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, ppo_rect)
        ppo_text = font.render("PPO auto", True, WHITE)
        screen.blit(ppo_text, (ppo_rect.x + 10, ppo_rect.y + 8))

        # GPT 버튼
        pygame.draw.rect(screen, BUTTON_HOVER if gpt_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, gpt_rect)
        gpt_text = font.render("GPT control", True, WHITE)
        screen.blit(gpt_text, (gpt_rect.x + 5, gpt_rect.y + 8))

        # 입력창 (GPT 모드일 때)
        pygame.draw.rect(screen, TEXTBOX_COLOR, input_box)
        input_surface = input_font.render(user_text, True, BLACK)
        screen.blit(input_surface, (input_box.x + 5, input_box.y + 5))

        # 확인 버튼
        pygame.draw.rect(screen, BUTTON_HOVER if submit_rect.collidepoint(mouse_x, mouse_y) else BUTTON_COLOR, submit_rect)
        submit_text = font.render("확인", True, WHITE)
        screen.blit(submit_text, (submit_rect.x + 10, submit_rect.y + 5))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if ppo_rect.collidepoint(event.pos):
                    shared_feedback["use_gpt_control"] = False
                    shared_feedback.pop("feedback", None)
                    print("[GUI] PPO auto mode activated")
                elif gpt_rect.collidepoint(event.pos):
                    shared_feedback["use_gpt_control"] = True
                    user_text = ""
                    typing = True
                    print("[GUI] GPT control mode activated. Please enter command:")
                elif submit_rect.collidepoint(event.pos) and typing:
                    typing = False
                    result = nl_controller.get_command(user_text)
                    shared_feedback["feedback"] = result
                    print(f"[GUI] Received GPT feedback: {result}")
            elif event.type == pygame.KEYDOWN and typing:
                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    user_text += event.unicode

        pygame.display.flip()
        clock.tick(10)
    pygame.quit()

def training_process(shared_feedback):
    config = dict(
        map=4,
        discrete_action=False,
        horizon=1000,
        random_spawn_lane_index=True,
        num_scenarios=10,
        traffic_density=0,
        accident_prob=0.1,
        log_level=50,
        use_render=True,
        force_render_fps=True,
        window_size=(1024, 768)
    )

    def make_env():
        env = LLMFeedbackControlWrapper(config, shared_feedback)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=128,
        learning_rate=1e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="mps"  # Mac M2 GPU (MPS) 사용
    )

    model.learn(total_timesteps=300_000, log_interval=4)
    model.save("ppo_metadrive_llm_feedback.zip")
    env.close()

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_feedback = manager.dict()
    shared_feedback["use_gpt_control"] = False

    train_proc = multiprocessing.Process(target=training_process, args=(shared_feedback,))
    gui_proc = multiprocessing.Process(target=gui_process, args=(shared_feedback,))

    train_proc.start()
    gui_proc.start()

    train_proc.join()
    gui_proc.join()
