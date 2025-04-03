import gymnasium as gym
import time
import collections

class TimeStampedQueueEnv(gym.Wrapper):
    def __init__(self, env, max_size=10000):
        super().__init__(env)
        self.queue = collections.deque(maxlen=max_size)
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # 큐에 현재 transition을 저장
        entry = {
            'timestamp': time.time(),   # 현재 시각
            'obs': self.last_obs,       # step 전 상태
            'action': action,
            'reward': reward
        }
        self.queue.append(entry)

        self.last_obs = obs
        self.get_queue()
        return obs, reward, done, truncated, info

    def get_near_states(self, query_time, tolerance=0.5):
        """
        query_time과 tolerance(초)를 기준으로, 큐에 저장된
        timestamp가 query_time ± tolerance 범위 내에 있는 항목을 반환
        """
        results = []
        for item in self.queue:
            if abs(item['timestamp'] - query_time) < tolerance:
                results.append(item)
        return results

    def get_queue(self):
        for entry in self.queue:
            print('timestamp : ',entry.get('timestamp'))
            print('action :', entry.get('action'))
            print('reward :',entry.get('reward'))
            print('\n')
        print('queue finish')
        