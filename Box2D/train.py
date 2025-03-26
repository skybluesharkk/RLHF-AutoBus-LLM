import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        # filename=None으로 설정하면 개별 로그 파일 생성을 방지할 수 있습니다.
        env = Monitor(env, filename=None)
        return env
    return _init

# 4개의 병렬 환경 생성
n_envs = 4
env = DummyVecEnv([make_env() for _ in range(n_envs)])

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

log_filename = "boxcar2d.txt"
log_file = open(log_filename, "w")
sys.stdout = log_file
model.learn(total_timesteps=1_000_000, log_interval=4)
model.save("ppo_boxcar2d.zip")
env.close()
