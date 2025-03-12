import sys
from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import multiprocessing
import time
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":

    log_filename = "my_train_test_log.txt"
    log_file = open(log_filename, "w")
    sys.stdout = log_file

    multiprocessing.set_start_method("spawn", force=True)  
    set_random_seed(0)

    #  환경 설정
    config = dict(
        map="C",
        discrete_action=False,
        horizon=500,
        random_spawn_lane_index=True,
        num_scenarios=3,
        traffic_density=0.1,
        accident_prob=0,
        log_level=50,
        use_render=False,               #  멀티스레드 렌더링 비활성화 (MacOS에서 충돌 가능)
        force_render_fps=True,        # FPS 제한 해제하여 부드러운 렌더링 유도
        window_size=(1024, 768)       #  윈도우 크기 변경 (기본 600x600이 문제될 수 있음)
    )


    #  병렬 환경 설정 (8개)
    num_envs = 8
    train_env = DummyVecEnv([lambda: Monitor(MetaDriveEnv(config))])

    #  PPO 모델 설정
    model = PPO("MlpPolicy",
                train_env,
                n_steps=2048,
                batch_size=128,
                learning_rate=1e-4,
                ent_coef=0.01,
                gamma=0.99,  
                gae_lambda=0.95, 
                verbose=1)

    #  모델 학습
    model.learn(total_timesteps=300_000, log_interval=4)
    
    #  모델 저장
    model.save("ppo_metadrive_test_mycode.zip")
    print(" Model saved: ppo_metadrive_test_mycode.zip")
    train_env.close()
    
    #  평가 환경 생성 (단일 환경)
    env = MetaDriveEnv(config)
    obs, _ = env.reset()

    total_reward = 0

    try:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            print(f"Step: {i}, Action: {action}, Reward: {reward}, Done: {done}")

            env.render()
            time.sleep(0.05)

            if done:
                total_reward = 0  
                obs, _ = env.reset()
        log_file.close()

        
        sys.stdout = sys.__stdout__
    finally:
        env.close()
