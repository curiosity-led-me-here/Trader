import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class WallStreet(gym.Env):
  def __init__(self, data, strike, feature_size=10, w_unrealised=0.3):       # (ttm_step, Strike, UL, C, P, PrevUL, PrevC, PrevP, CumUnrealisedPnL, CumRealisedPnL)
    super().__init__()
    self.unrealised_w = w_unrealised
    self.data = data
    self.strike_price = strike
    self.feature_size = feature_size
    self.cash = 0
    self.prev_rew = 0
    self.positions = []
    self.unrealised = 0
    self.current_step = 0
    self.action_space = spaces.Discrete(5)        # (Buy Call, Sell Call, Buy Put, Sell Put, Hold)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, feature_size), dtype=np.float32)
    self.current_state = None
    self.reset()

  def reset(self):
    state = np.zeros(self.feature_size, dtype=np.float32)
    self.current_step = 0
    self.cash = 0
    self.prev_rew = 0
    self.unrealised = 0
    state[1] = self.strike_price
    self.current_state = np.array(self.current_state, dtype=np.float32)
    self.current_state = state
    return self.current_state, {}

  def step(self, action):
# updating current step
    self.current_step += 1
# episode termination logic
    if self.current_step >= len(self.data):
      if any(pos[0] == 'CALL' for pos in self.positions):
        for pos in self.positions[:]:
          if pos[0] == 'CALL':
            reward = float(self.data.iloc[self.current_step-1].iloc[2]) - pos[1]
            self.cash += reward
            self.positions.remove(pos)
      if any(pos[0] == 'PUT' for pos in self.positions):
        for pos in self.positions[:]:
          if pos[0] == 'PUT':
            reward = float(self.data.iloc[self.current_step-1].iloc[3]) - pos[1]
            self.cash += reward
            self.positions.remove(pos)
      self.current_state[0] = self.current_step
      reward = self.cash
      return self.current_state, reward, True, {}
    reward = 0
# ttm updation
    self.current_state[0] = self.current_step
# daily data extraction
    if self.current_step != 0:
      prev_day = self.data.iloc[self.current_step-1].to_numpy()
      today = self.data.iloc[self.current_step].to_numpy()
    else:
      today = self.data.iloc[self.current_step].to_numpy()
# current state price updation
    for i in range(len(today)-1):
      self.current_state[i+2] = today[i+1]
      if self.current_state[0] != 0:
        self.current_state[i+5] = prev_day[i+1]
      else:
        self.current_state[i+5] = 0
# action-based variable updation
    txn_cost = 5
    error_order_penalty = 100
    if action == 0:
      if not any(pos[0] == 'CALL' for pos in self.positions) and not any(pos[0] == 'PUT' for pos in self.positions):
        self.positions.append(('CALL', float(today[2])))
      else:
        self.cash -= error_order_penalty
        self.current_state[9] = self.cash
        for i in range(len(self.positions)):
          if self.positions[i][0] == 'CALL':
            self.unrealised = today[2] - self.positions[i][1]
            self.current_state[8] = self.unrealised
          elif self.positions[i][0] == 'PUT':
            self.unrealised = today[3] - self.positions[i][1]
            self.current_state[8] = self.unrealised
    elif action == 1:
      if any(pos[0] == 'CALL' for pos in self.positions):
        for pos in self.positions[:]:
          if pos[0] == 'CALL':
            reward = today[2] - pos[1]
            self.cash += reward - txn_cost
            self.positions.remove(pos)
            self.current_state[9] = self.cash
            self.current_state[8] = 0.0
            #break
        else:
          self.cash -= error_order_penalty
          self.current_state[9] = self.cash
        for i in range(len(self.positions)):
          if self.positions[i][0] == 'CALL':
            self.unrealised = today[2] - self.positions[i][1]
            self.current_state[8] = self.unrealised
          elif self.positions[i][0] == 'PUT':
            self.unrealised = today[3] - self.positions[i][1]
            self.current_state[8] = self.unrealised
    elif action == 2:
      if not any(pos[0] == 'CALL' for pos in self.positions) and not any(pos[0] == 'PUT' for pos in self.positions):
        self.positions.append(('PUT', float(today[3])))
      else:
        self.cash -= error_order_penalty
        self.current_state[9] = self.cash
        for i in range(len(self.positions)):
          if self.positions[i][0] == 'CALL':
            self.unrealised = today[2] - self.positions[i][1]
            self.current_state[8] = self.unrealised
          elif self.positions[i][0] == 'PUT':
            self.unrealised = today[3] - self.positions[i][1]
            self.current_state[8] = self.unrealised
    elif action == 3:
      if any(pos[0] == 'PUT' for pos in self.positions):
        for pos in self.positions[:]:
          if pos[0] == 'PUT':
            reward = today[3] - pos[1]
            self.cash += reward - txn_cost
            self.positions.remove(pos)
            self.current_state[9] = self.cash
            self.current_state[8] = 0.0
            #break
      else:
        self.cash -= error_order_penalty
        self.current_state[9] = self.cash
        for i in range(len(self.positions)):
          if self.positions[i][0] == 'CALL':
            self.unrealised = today[2] - self.positions[i][1]
            self.current_state[8] = self.unrealised
          elif self.positions[i][0] == 'PUT':
            self.unrealised = today[3] - self.positions[i][1]
            self.current_state[8] = self.unrealised
    elif action == 4:
      for i in range(len(self.positions)):
        if self.positions[i][0] == 'CALL':
          self.unrealised = today[2] - self.positions[i][1]
          self.current_state[8] = self.unrealised
        elif self.positions[i][0] == 'PUT':
          self.unrealised = today[3] - self.positions[i][1]
          self.current_state[8] = self.unrealised
    final_reward = (self.unrealised * self.unrealised_w) + self.cash
    self.current_state = [float(x) for x in self.current_state]
    return self.current_state, final_reward, False, {}