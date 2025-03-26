import gymnasium as gym
import pygame
import numpy as np
import time

def get_action_from_keys(keys):
    """
    pygame의 키 입력을 CarRacing/BoxCar2D 환경의 연속 행동 벡터로 매핑합니다.
    행동 공간: [steering, gas, brake]
      - steering: -1 (왼쪽) ~ 1 (오른쪽)
      - gas: 0 ~ 1
      - brake: 0 ~ 1
    """
    steering = 0.0
    gas = 0.0
    brake = 0.0
    if keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_RIGHT]:
        steering = 1.0
    if keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_DOWN]:
        brake = 1.0
    return np.array([steering, gas, brake], dtype=np.float32)

def collect_demo_data(env_name="CarRacing-v3", total_steps=5000, demo_file="demo_data.npz"):
    """
    데모 데이터를 수집합니다.
      - env_name: 사용할 환경 이름. BoxCar2D 환경이 있다면 "BoxCar2D-v0" 등으로 변경.
      - total_steps: 총 몇 스텝의 데이터를 수집할지 지정.
      - demo_file: 수집된 데이터를 저장할 파일 이름.
    """
    # render_mode="human"로 설정하면, 환경이 pygame 창을 통해 화면에 표시됩니다.
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()
    
    demo_observations = []
    demo_actions = []
    
    # pygame 초기화 및 입력 창 설정
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("데모 데이터 수집 - Q키를 눌러 종료")
    
    clock = pygame.time.Clock()
    step = 0
    running = True
    print("데모 데이터 수집 시작. 좋은 운전 예시를 보여주세요. (종료: Q키)")
    
    while running and step < total_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Q키가 눌리면 종료
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        keys = pygame.key.get_pressed()
        action = get_action_from_keys(keys)
        
        # 환경 step 진행
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 관측과 행동 저장 (행동 클로닝에서는 observation, action 쌍만 사용)
        demo_observations.append(obs)
        demo_actions.append(action)
        
        obs = next_obs
        step += 1
        
        clock.tick(60)  # 60 FPS로 실행하여 입력 반응성을 유지
        
        if done or truncated:
            obs, info = env.reset()
    
    env.close()
    pygame.quit()
    
    # 리스트를 NumPy 배열로 변환 후 저장
    demo_observations = np.array(demo_observations, dtype=np.float32)
    demo_actions = np.array(demo_actions, dtype=np.float32)
    np.savez(demo_file, observations=demo_observations, actions=demo_actions)
    print(f"데모 데이터 수집 종료. {demo_file} 파일에 저장되었습니다. (총 스텝: {step})")

if __name__ == "__main__":
    collect_demo_data()
