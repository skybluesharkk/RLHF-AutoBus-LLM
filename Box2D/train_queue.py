from timeStampQueue import  TimeStampedQueueEnv
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import Monitor


base_env = gym.make("CarRacing-v3", render_mode="rgb_array")
queue_env = TimeStampedQueueEnv(base_env)
queue_env = Monitor(queue_env, filename="monitor_log_q.txt")

def linear_schedule(initial_value):
    def func(progress_remaining):
        # progress_remaining: 학습 시작시 1에서 종료시 0으로 선형 감소
        return progress_remaining * initial_value
    return func

model = PPO(
    "CnnPolicy",
    queue_env,
    n_steps=1024,  
    batch_size=128,
    learning_rate=linear_schedule(3e-4), 
    clip_range=0.1,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    device="mps"
)


log_filename = "boxcar2d_q.txt"
log_file = open(log_filename, "w")
sys.stdout = log_file
model.learn(total_timesteps=1_000_000, log_interval=4)
model.save("ppo_boxcar2d_q.zip")
queue_env.close()

