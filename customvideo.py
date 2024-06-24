import gym
from gym import spaces
import numpy as np
import cv2


class CustomVideoEnv(gym.Env):
    def __init__(self, video_path, relevance_function, frame_size=(64, 64)):
        super(CustomVideoEnv, self).__init__()
        
        self.video_path = video_path
        self.relevance_function = relevance_function
        self.frame_size = frame_size
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(frame_size[0], frame_size[1], 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Normalized coordinates (x, y)
        
        self.current_frame = None
        self.current_frame_index = 0
    
    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_index = 0
        self.current_frame = self._get_next_frame()
        return self._get_observation(np.array([0.5, 0.5]))  # Start by looking at the center
    
    def step(self, action):
        done = False
        self.current_frame_index += 1
        
        if self.current_frame_index >= self.total_frames:
            done = True
            return self._get_observation(action), 0, done, {}
        
        self.current_frame = self._get_next_frame()
        obs = self._get_observation(action)
        
        relevance = self.relevance_function(obs)
        reward = relevance
        
        return obs, reward, done, {}
    
    def _get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame
    
    def _get_observation(self, action):
        h, w, _ = self.current_frame.shape
        x = int(action[0] * w)
        y = int(action[1] * h)
        
        x_start = max(0, x - self.frame_size[0] // 2)
        y_start = max(0, y - self.frame_size[1] // 2)
        
        x_end = min(w, x_start + self.frame_size[0])
        y_end = min(h, y_start + self.frame_size[1])
        
        obs = self.current_frame[y_start:y_end, x_start:x_end]
        
        if obs.shape[0] != self.frame_size[0] or obs.shape[1] != self.frame_size[1]:
            obs = cv2.resize(obs, self.frame_size)
        
        return obs

    def render(self, mode='human'):
        if self.current_frame is not None:
            cv2.imshow('Current Frame', self.current_frame)
            cv2.waitKey(1)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()