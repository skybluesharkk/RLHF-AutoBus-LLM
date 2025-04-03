import time
from collections import deque

class Event:
    """
    - timestamp: 이벤트가 발생한 시각(time.time() 기준)
    - obs: 관측(예: 화면 스냅샷 또는 state)
    - reward: 원본 환경에서 받은 보상(추후 사용자 피드백으로 수정 가능)
    - action: 에이전트가 취한 action
    """
    def __init__(self, timestamp, obs, reward, action):
        self.timestamp = timestamp
        self.obs = obs
        self.reward = reward
        self.action = action

class EventQueue:
    """
    - max_duration: 큐 안에서 보관할 이벤트의 최대 유지 시간(초 단위)
    """
    def __init__(self, max_duration=5.0):
        self.max_duration = max_duration
        self.queue = deque()
    
    def push(self, event: Event):
        """
        새로운 이벤트를 큐에 추가하고,
        큐 맨 앞에 있는 '너무 오래된' 이벤트는 제거한다.
        """
        self.queue.append(event)
        self._remove_old_events()
    
    def _remove_old_events(self):
        """
        현재 시각으로부터 max_duration보다 오래된 이벤트는 큐에서 제거
        """
        current_time = time.time()
        while self.queue and (current_time - self.queue[0].timestamp) > self.max_duration:
            self.queue.popleft()
    
    def find_event_by_time(self, timestamp, tolerance=0.05):
        """
        특정 시각(timestamp) 근방(tolerance) 내의 이벤트를 찾는다.
        - tolerance=0.05(50ms) 정도로 유연하게 잡아 동일 시점이라고 간주
        - 실제 게임 로직이나 사용자 입력 타이밍에 따라 조정 가능
        """
        # 여기서는 가장 최근 것부터 탐색하는 예시
        for event in reversed(self.queue):
            if abs(event.timestamp - timestamp) <= tolerance:
                return event
        return None
def process_user_feedback(event_queue: EventQueue, user_input: str, input_timestamp: float):
    """
    - user_input: 'p' 혹은 'n' 등
    - input_timestamp: 사용자가 입력한 시점(대략 time.time()으로 찍음)
    """
    event = event_queue.find_event_by_time(input_timestamp)
    if event is None:
        print(f"[피드백 처리] {input_timestamp} 근처의 이벤트를 찾을 수 없습니다.")
        return
    
    # 사용자 피드백에 따라 보상을 수정(+- 1). 필요시 가중치를 다르게 줄 수도 있음
    if user_input == 'p':
        event.reward += 1.0
        print(f"[피드백 처리] 긍정 보상 +1 -> 수정된 보상: {event.reward}")
    elif user_input == 'n':
        event.reward -= 1.0
        print(f"[피드백 처리] 부정 보상 -1 -> 수정된 보상: {event.reward}")
    else:
        print(f"[피드백 처리] 알 수 없는 명령어: {user_input}")
import gymnasium as gym
import time
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 위에서 정의한 Event, EventQueue, process_user_feedback 임포트했다고 가정 ---

class CarRacingWrapper(gym.Wrapper):
    """
    CarRacing 환경에서 step할 때마다 EventQueue에 이벤트를 기록하고,
    수정된 보상을 활용해 학습 가능한 Wrapper 예시.
    """
    def __init__(self, env, event_queue: EventQueue):
        super(CarRacingWrapper, self).__init__(env)
        self.event_queue = event_queue
    
    def step(self, action):
    # gym 버전이 0.26 이상인 경우
    # obs, reward, done, truncated, info = self.env.step(action)
    # done = done or truncated

    # (아직 구 버전 환경이면, 기존처럼 4개만 받을 수 있게)
    # obs, reward, done, info = self.env.step(action)
    
    # 현재 CarRacing-v3에서는 5개 반환하므로 다음처럼 받습니다:
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or truncated
        
        current_time = time.time()
        event = Event(
            timestamp=current_time, 
            obs=obs, 
            reward=reward, 
            action=action
        )
        self.event_queue.push(event)

        updated_reward = event.reward
        return obs, updated_reward, done, info


def user_input_listener(event_queue: EventQueue):
    """
    별도 스레드에서 사용자 입력을 상시 모니터링하여
    p 또는 n 명령이 들어오면 이벤트 큐의 해당 시점 보상을 수정한다.
    """
    while True:
        user_cmd = input("사용자 피드백(p/n)을 입력하세요: ")
        timestamp = time.time()
        if user_cmd in ['p', 'n']:
            process_user_feedback(event_queue, user_cmd, timestamp)
        else:
            print(f"알 수 없는 명령: {user_cmd}")

def main():
    # 1) 이벤트 큐 생성
    event_queue = EventQueue(max_duration=5.0)
    
    # 2) CarRacing 환경 래핑
    env = gym.make('CarRacing-v3')  # (버전에 맞게 이름 조정)
    env = CarRacingWrapper(env, event_queue)
    env = DummyVecEnv([lambda: env])

    # 3) PPO 모델 생성
    model = PPO("CnnPolicy", env, verbose=1)

    # 4) 사용자 입력을 모니터링할 스레드 시작
    input_thread = threading.Thread(target=user_input_listener, args=(event_queue,), daemon=True)
    input_thread.start()
    
    # 5) 학습 루프
    #    - 각 learn 사이클마다 env에서 step()이 일어나고,
    #      이벤트가 큐에 쌓이며, 사용자는 p/n 명령으로 보상 수정 가능
    #    - PPO는 내부적으로 rollout 수집 후 학습 업데이트를 수행
    #      -> step에서 이미 수정된 보상을 가져오기 때문에 자연스럽게 반영됨
    for _ in range(100):  # 예: 100번 반복
        model.learn(total_timesteps=2048)
        # 필요시 중간에 queue를 살펴보거나 로깅할 수 있음

    # 종료
    env.close()

if __name__ == "__main__":
    main()
