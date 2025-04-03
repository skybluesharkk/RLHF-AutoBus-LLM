import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 1) 간단한 피드백 소스 예시:
# 실제로는 별도 큐, GUI, human feedback 등을 통해 수집 가능
class SimpleFeedbackSource:
    """
    transition의 timestamp와 비교해, ±0.5초 이내면 new_reward로 교체 (예시).
    feedback_data: [{"timestamp": T, "new_reward": R}, ...]
    """
    def __init__(self, feedback_data, tolerance=0.5):
        self.feedback_data = feedback_data
        self.tolerance = tolerance

    def get_reward_for_timestamp(self, ts):
        # feedback_data를 순회해, |T - ts| < tolerance 인 항목 찾아 반환
        for fb in self.feedback_data:
            if abs(fb["timestamp"] - ts) < self.tolerance:
                return fb["new_reward"]
        return None


# 2) 후처리 콜백 구현
class PostprocessRolloutCallback(BaseCallback):
    """
    rollout이 끝나면(self.model.rollout_buffer 사용),
    info["timestamp"]를 확인해, feedback_source에서 보상 수정.
    """
    def __init__(self, feedback_source, verbose=0):
        super().__init__(verbose)
        self.feedback_source = feedback_source
    def _on_step(self) -> bool:
        """
        SB3가 매 스텝(step)마다 호출하는 추상 메서드.
        여기서는 아무 것도 안 해도 괜찮으므로 True만 리턴.
        """
        return True
    def _on_rollout_end(self) -> None:
        rb = self.model.rollout_buffer
        # rollout 버퍼 크기
        buffer_size = len(rb.observations)
        # infos[i]에 timestamp 있다고 가정
        for i in range(buffer_size):
            info_i = rb.infos[i]
            if info_i is not None:
                ts = info_i.get("timestamp", None)
                if ts is not None:
                    new_reward = self.feedback_source.get_reward_for_timestamp(ts)
                    if new_reward is not None:
                        old_reward = rb.rewards[i]
                        rb.rewards[i] = new_reward
                        if self.verbose:
                            print(f"[Callback] idx={i}, ts={ts:.2f}, reward {old_reward} -> {new_reward}")

        if self.verbose:
            print("PostprocessRolloutCallback done: rollout buffer updated with human feedback.")


# 3) 환경 래퍼: info["timestamp"] 기록
class TimestampBox2DEnv(gym.Wrapper):
    """
    예: CarRacing-v3 등 Box2D 환경에서
    step() 시 info["timestamp"] = time.time()
    """
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info["timestamp"] = time.time()
        return obs, reward, done, truncated, info

def main():
    # 예시 feedback 데이터
    # 실제론 GUI나 파일에서 ± 시점에 "timestamp"와 "new_reward"를 수집
    example_feedback = [
        {"timestamp": time.time() + 5,  "new_reward": -10.0}, # 5초 후 전이 보상 -10
        {"timestamp": time.time() + 10, "new_reward": +5.0},  # 10초 후 전이 보상 +5
    ]
    feedback_source = SimpleFeedbackSource(example_feedback, tolerance=0.5)

    # CarRacing-v3 예시 (Box2D)
    def make_env():
        base_env = gym.make("CarRacing-v3", render_mode="rgb_array") # 또는 "human"
        wrapped_env = TimestampBox2DEnv(base_env)
        return Monitor(wrapped_env)

    env = DummyVecEnv([make_env])

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
    )

    # 콜백 등록
    callback = PostprocessRolloutCallback(feedback_source=feedback_source, verbose=1)

    # 학습: rollout 수집 끝날 때마다 콜백이 호출돼 보상이 수정됨
    model.learn(total_timesteps=3000, callback=callback)

    model.save("ppo_box2d_feedback.zip")
    env.close()
    print("Training done!")

if __name__ == "__main__":
    main()
