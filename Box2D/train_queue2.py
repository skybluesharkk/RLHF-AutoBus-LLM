import time
import gymnasium as gym
from timeStampQueue import TimeStampedQueueEnv  # 실제 모듈명으로 변경하세요

# 환경 초기화
base_env = gym.make("CarRacing-v3", render_mode="rgb_array")
queue_env = TimeStampedQueueEnv(base_env)

# 환경 리셋
obs, info = queue_env.reset()

# 몇 번 step 수행 (데이터 확보용)
for _ in range(10):
    action = base_env.action_space.sample()
    obs, reward, done, truncated, info = queue_env.step(action)
    time.sleep(0.2) 
# 사용자 입력 대기
input("조회하려는 시점에 Enter")

# 현재 시각 기록
query_time = time.time()

# 해당 시점 주변의 transition 검색
nearby = queue_env.get_near_states(query_time, tolerance=0.5)

# 결과 출력
print("\n조회된 Transition들")
if nearby:
    for item in nearby:
        print("Timestamp:", item['timestamp'])
        print("Action:", item['action'])
        print("Reward:", item['reward'])
        print("------")
else:
    print("조회 시간 ±0.5초 범위 내의 transition이 없습니다.")
