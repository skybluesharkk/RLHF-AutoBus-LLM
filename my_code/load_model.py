import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor


model_path = "ppo_metadrive_test_mycode"
model = PPO.load(model_path)
print(f"Loaded model from {model_path}")

env = MetaDriveEnv(dict(use_render=True,
                        map=4))
num_episodes = 10  
episode_rewards = []
episode_lengths = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    while True:
        #  모델이 연속적인 action을 반환하므로 환경에 맞게 변환 필요
        action, _states = model.predict(obs, deterministic=True)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(action)  # 단일 이산 행동 변환
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action = action.astype(int)  # 여러 개의 이산 행동 변환

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # 렌더링 (시각적으로 확인)
        env.render(mode="human",map=4)
        time.sleep(0.05)  # 속도 조절

        if done:
            print(f"🎯 Episode {ep+1} finished - Total Reward: {total_reward}, Steps: {step_count}")
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            break

# 평가 결과 출력
avg_reward = np.mean(episode_rewards)
avg_steps = np.mean(episode_lengths)

print("\n**Evaluation Summary**")
print(f"Number of Episodes: {num_episodes}")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Average Episode Length: {avg_steps:.2f}")

# 환경 종료
env.close()
