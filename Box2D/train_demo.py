import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 1. 데모 데이터 로드 
# demo_data.npz 파일은 'observations'와 'actions' 배열을 포함해야 함.
def load_demo_data(file_path='demo_data.npz'):
    data = np.load(file_path)
    observations = data['observations']  # 예: (N, H, W, C)
    actions = data['actions']            # 예: (N, action_dim)
    print(f"Loaded {observations.shape[0]} demonstration samples.")
    return observations, actions


# 2. 행동 클로닝(Behavior Cloning) 사전학습 함수
def pretrain_policy(model, dataloader, epochs=5, lr=1e-4, device="mps"):
    model.policy.to(device)
    optimizer = optim.Adam(model.policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.policy.train()  # 학습 모드

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            # PPO의 CnnPolicy는 (batch, channels, height, width) 형태를 요구.
            batch_obs = batch_obs.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            dist = model.policy.get_distribution(batch_obs)

            pred_actions = dist.mode()  
            loss = mse_loss(pred_actions, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Pretraining] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    model.policy.eval()  # 평가 모드로 전환



# 3. 환경 생성 
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = Monitor(env, filename=None)
        return env
    return _init

n_envs = 4  # 병렬 환경의 개수
env = DummyVecEnv([make_env() for _ in range(n_envs)])

# 4. PPO 모델 생성

model = PPO(
    "CnnPolicy",
    env,
    n_steps=1024,
    batch_size=128,
    learning_rate=3e-4,
    clip_range=0.1,
    ent_coef=0.02,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    device="mps"
)

# 5. 데모 데이터를 사용한 사전학습 (행동 클로닝)
observations, actions = load_demo_data("demo_data.npz")
# 데모 데이터는 float32 타입으로 변환
observations = observations.astype(np.float32)
actions = actions.astype(np.float32)
# PyTorch 텐서로 변환
observations_tensor = torch.tensor(observations)
actions_tensor = torch.tensor(actions)
dataset = TensorDataset(observations_tensor, actions_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print("행동 클로닝 사전학습 시작")
pretrain_policy(model, dataloader, epochs=5, lr=1e-4, device="mps")
print("사전학습 완료")


model.learn(total_timesteps=1_000_000, log_interval=4)
model.save("ppo_boxcar2d_d.zip")
env.close()

print("모델 학습 및 저장 완료")
