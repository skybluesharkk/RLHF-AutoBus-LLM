import time
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ------------------------
# 1. 커스텀 콜백 정의
# ------------------------
class TimeStampedTransitionCallback(BaseCallback):
    """
    매 스텝마다 RolloutBuffer에서 (obs, action, reward) 등을 꺼내
    time.time()과 함께 별도 리스트(self.transitions)에 저장합니다.

    이후 사용자가 특정 시점(query_time)을 지정하면,
    그와 '가까운' 타임스탬프를 가진 트랜지션을 조회할 수 있습니다.
    """
    def __init__(self, tolerance=1.0, verbose=0):
        super().__init__(verbose)
        self.tolerance = tolerance
        self.transitions = []  # (timestamp, obs, action, reward) 등을 저장

    def _on_step(self) -> bool:
        """
        매 환경 스텝마다 호출되는 메서드로,
        RolloutBuffer의 마지막 transition 정보를 가져와 기록합니다.
        """
        # (주의) rollout_buffer.pos는 다음 기록 위치를 가리키므로, -1 하면 '현재 스텝'이 됩니다.
        pos = self.model.rollout_buffer.pos - 1
        if pos < 0:
            # 인덱스가 -1이면 실제로는 buffer 맨 끝(= buffer_size - 1)을 의미
            pos = self.model.rollout_buffer.buffer_size - 1

        # --- RolloutBuffer에서 관측, 행동, 보상을 꺼내기 ---
        # 1) obs, action이 torch.Tensor이므로 detach().cpu().numpy() 형태로 변환
        obs = self.model.rollout_buffer.observations[pos].detach().cpu().numpy()
        action = self.model.rollout_buffer.actions[pos].detach().cpu().numpy()
        reward = float(self.model.rollout_buffer.rewards[pos].item())

        # 2) 현재 시각 기록
        timestamp = time.time()

        # 3) transition 리스트에 추가
        self.transitions.append({
            "timestamp": timestamp,
            "obs": obs,
            "action": action,
            "reward": reward
        })

        return True  # 학습 계속 진행

    def query_near_time(self, query_time: float):
        """
        사용자가 지정한 'query_time'과 절댓값 기준으로
        'tolerance' 이내에 속하는 트랜지션을 필터링해 반환합니다.
        """
        return [
            t for t in self.transitions
            if abs(t["timestamp"] - query_time) < self.tolerance
        ]


# -------------------------
# 2. 기타 함수(스케줄러)
# -------------------------
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


# -------------------------
# 3. 메인 실행부
# -------------------------
if __name__ == "__main__":
    # 환경 준비
    base_env = gym.make("CarRacing-v3", render_mode="rgb_array")
    base_env = Monitor(base_env)  # 모니터 래퍼
    vec_env = DummyVecEnv([lambda: base_env])

    # 모델 정의
    model = PPO(
        "CnnPolicy",
        vec_env,
        n_steps=1024,
        batch_size=128,
        learning_rate=linear_schedule(3e-4),
        clip_range=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        device="mps",  
    )

    # 콜백 준비 (timestamp 기록용)
    callback = TimeStampedTransitionCallback(tolerance=1.0)


    steps_per_iter = 2000
    total_timesteps = 10000

    # 총 (10000/2000) = 5번 반복
    n_iters = total_timesteps // steps_per_iter
    for iteration in range(n_iters):
        # 매 구간마다 model.learn 실행
        model.learn(
            total_timesteps=steps_per_iter,
            reset_num_timesteps=False,
            callback=callback
        )
        print(f"\n[구간 {iteration+1}/{n_iters}] 학습 완료.\n")

        # -----------------------------
        # 2) 사용자 입력 받기
        # -----------------------------
        user_input = input("입력 대기 - 'p' 입력 시 지금 시간대의 Transition 조회, 'q' 입력 시 종료, 엔터(그냥 Enter)면 계속: ")
        if user_input.strip().lower() == 'p':
            # p 입력 시, 현재 시간을 기준으로 주변 타임스탬프를 갖는 트랜지션 조회
            query_time = time.time()
            near_transitions = callback.query_near_time(query_time)

            print(f"▶ 현재 시각({query_time}) ± {callback.tolerance}초 이내의 Transition: {len(near_transitions)}개")
            for i, t in enumerate(near_transitions[:5]):  # 앞 5개만 예시로 출력
                print(f"  [{i}] time: {t['timestamp']}, action: {t['action']}, reward: {t['reward']}")
            print()

        elif user_input.strip().lower() == 'q':
            # 학습 중단
            print("중단")
            break

    # 마지막으로 모델 저장
    model.save("ppo_car_racing_with_timestamped_transitions")

