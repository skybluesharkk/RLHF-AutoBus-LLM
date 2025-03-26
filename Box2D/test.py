import gymnasium as gym
from stable_baselines3 import PPO

# 학습 완료된 모델 로드
model = PPO.load("ppo_boxcar2d.zip")

# 시각화를 위해 render_mode를 "human"으로 설정하여 환경 생성
env = gym.make("CarRacing-v3", render_mode="human")
observation, info = env.reset()

while True:
    # 모델의 예측을 사용해 액션 결정 (deterministic=True로 설정하여 결정론적으로 행동)
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, truncated, info = env.step(action)
    
    # 환경 렌더링 (이미 "human" 모드라 자동으로 출력되지만, 호출해도 무방)
    env.render()
    
    # 에피소드 종료 시 환경 재설정
    if done or truncated:
        observation, info = env.reset()
