import time
import collections
import numpy as np
import pygame
import torch as th
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

# =======================
# A. HumanFeedbackInterface via Button Click
# =======================
class HumanFeedbackInterface:
    """
    버튼 클릭으로 보상을 조정:
    - 왼쪽 버튼(Rect) 클릭 => +1 보상
    - 오른쪽 버튼(Rect) 클릭 => -1 보상
    """
    def __init__(self, tolerance=0.5):
        self.queue = []
        self.tolerance = tolerance

        # 버튼 Rect 정의 (x, y, w, h)
        self.plus_button = pygame.Rect(50, 50, 80, 40)   # +1 보상
        self.minus_button = pygame.Rect(150, 50, 80, 40) # -1 보상

        # 글꼴(버튼 텍스트 표시용)
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)

    def draw_buttons(self, screen):
        """
        버튼(사각형)과 텍스트를 그리는 함수.
        매 프레임마다 호출.
        """
        # 플러스 버튼
        pygame.draw.rect(screen, (0, 200, 0), self.plus_button)  # 녹색
        plus_text = self.font.render("+1", True, (255, 255, 255))
        screen.blit(plus_text, (self.plus_button.x+20, self.plus_button.y+5))

        # 마이너스 버튼
        pygame.draw.rect(screen, (200, 0, 0), self.minus_button) # 빨강
        minus_text = self.font.render("-1", True, (255, 255, 255))
        screen.blit(minus_text, (self.minus_button.x+20, self.minus_button.y+5))

    def poll_feedback(self, screen):
        """
        - pygame 이벤트에서 MOUSEBUTTONDOWN 감지
        - 버튼 영역 클릭 시 보상 +1 또는 -1
        - (timestamp, delta_reward)를 queue에 저장
        """
        # 먼저 버튼들 그리기(매 프레임 호출)
        self.draw_buttons(screen)

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                timestamp = time.time()
                mouse_pos = pygame.mouse.get_pos()
                if self.plus_button.collidepoint(mouse_pos):
                    print("[Feedback] +1 Button clicked")
                    self.queue.append({"timestamp": timestamp, "delta_reward": +1.0})
                elif self.minus_button.collidepoint(mouse_pos):
                    print("[Feedback] -1 Button clicked")
                    self.queue.append({"timestamp": timestamp, "delta_reward": -1.0})

        # queue의 내용을 반환하고 내부 큐를 비움
        feedback_events = self.queue.copy()
        self.queue.clear()
        return feedback_events


# =======================
# B. TimeStampedQueueEnv
# =======================
class TimeStampedQueueEnv(gym.Wrapper):
    """
    - CarRacing 환경 래퍼
    - env.step()시 info["timestamp"] = time.time() 저장
    - 내부 큐에 전이를 보관 -> pop_transition()으로 수거
    """
    def __init__(self, env, max_size=1000):
        super().__init__(env)
        self.transition_queue = collections.deque(maxlen=max_size)
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_time = time.time()
        info["timestamp"] = current_time

        entry = {
            "timestamp": current_time,
            "obs_before": self.last_obs,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }
        self.transition_queue.append(entry)
        self.last_obs = obs
        return obs, reward, done, truncated, info

    def pop_transition(self):
        if len(self.transition_queue) > 0:
            return self.transition_queue.popleft()
        return None

# =======================
# C. FeedbackPPO
# =======================
class FeedbackPPO(PPO):
    """
    - collect_rollouts() 오버라이딩
    - 매 스텝마다 TimeStampedQueueEnv에서 전이 pop
    - HumanFeedbackInterface.poll_feedback() -> 버튼 클릭 이벤트
    - 해당 timestamp 근접이면 보상 수정
    - rollout buffer에 최종 add
    """
    def __init__(self, *args, human_feedback=None, feedback_tolerance=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.human_feedback = human_feedback
        self.feedback_tolerance = feedback_tolerance

    def collect_rollouts(self, env: GymEnv, callback: MaybeCallback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        # DummyVecEnv(n_envs=1) 전제
        # CarRacing render_mode="human" => pygame display
        # human_feedback.draw_buttons()를 위해 screen 객체를 얻을 필요가 있음
        # SB3가 별도 pygame display를 갖고 있지 않으므로,
        # pygame.display.get_surface() 등으로 screen 얻기
        screen = pygame.display.get_surface()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()
            # CarRacing continuous -> clip
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            n_steps += 1

            # TimeStampedQueueEnv 전이 pop
            transition = env.envs[0].pop_transition()
            if transition is None:
                pass
            else:
                updated_reward = transition["reward"]

                # 버튼 이벤트 폴링(버튼 그리기 & 클릭 체크)
                # -> 여러 번 클릭되면 모두 합산
                feedback_events = self.human_feedback.poll_feedback(screen) if self.human_feedback else []
                for fb in feedback_events:
                    fb_time = fb["timestamp"]
                    fb_value = fb["delta_reward"]
                    if abs(transition["timestamp"] - fb_time) < self.feedback_tolerance:
                        print(f"[FeedbackPPO] add {fb_value} to original reward {updated_reward}")
                        updated_reward += fb_value

                # rollout_buffer.add
                done_bool = dones[0]
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    np.array([updated_reward], dtype=np.float32),
                    self._last_episode_starts,
                    values,
                    log_probs
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones

            # 속도 늦춰서 클릭 여유
            time.sleep(0.05)

            # pygame 화면 업데이트
            # CarRacing이 이미 그려주지만, 버튼 텍스트 갱신 위해 flip() 호출
            pygame.display.flip()

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))  # 버튼 표시 위해 좀 큰 화면
    pygame.display.set_caption("CarRacing with Buttons Feedback")

    print("Click +1 / -1 buttons to modify reward")

    # CarRacing (render=human)
    base_env = gym.make("CarRacing-v3", render_mode="human")
    monitored_env = Monitor(base_env)
    time_env = TimeStampedQueueEnv(monitored_env)

    env = DummyVecEnv([lambda: time_env])  # n_envs=1

    # 버튼 클릭 기반 Feedback Interface
    human_feedback = HumanFeedbackInterface(tolerance=0.5)

    # PPO 모델
    model = FeedbackPPO(
        "CnnPolicy",
        env,
        n_steps=512,
        batch_size=128,
        learning_rate=3e-4,
        clip_range=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="mps",
        human_feedback=human_feedback,
        feedback_tolerance=0.5
    )

    model.learn(total_timesteps=10_000, log_interval=1)
    model.save("ppo_feedback_buttons.zip")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
