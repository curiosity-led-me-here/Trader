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
from processing.converter import split

path = r'/Users/ashu/Documents/Trader/NIFTY 50/output_.xlsx'
data = pd.read_excel(path)
input_data = split(data)
print(input_data)


class Wallstreet(gym.Env):
    def __init__(self, dataframes, transaction_cost=0.01):
        super().__init__()
        self.dataframes = dataframes   # list of option chain snapshots
        self.transaction_cost = transaction_cost
        self.cum_pnl = 0
        
        # fixed strikes across all timesteps
        self.strikes = sorted(set.union(*[set(df['strike']) for df in dataframes]))
        self.strike_to_idx = {k: i for i, k in enumerate(self.strikes)}
        
        # two types: Call (0), Put (1)
        self.n_strikes = len(self.strikes)
        self.n_types = 2
        
        # action space = strike index + type, or flat
        self.action_space = gym.spaces.Discrete(self.n_strikes * self.n_types + 1)
        
        # observation: (n_strikes, 2, features)
        # features = [lnMoneyness, price]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_strikes, self.n_types, 2), 
            dtype=np.float32
        )
        
    def reset(self):
        self.t = 0
        self.position = None  # (strike_idx, type_idx) or None
        return self._get_obs()
    
    def step(self, action):
        done = False
        reward = 0.0
        prev_position = self.position
        
        # decode action
        if action == self.n_strikes * self.n_types:
            self.position = None  # flat
        else:
            strike_idx = action // self.n_types
            type_idx = action % self.n_types
            self.position = (strike_idx, type_idx)
        
        # compute reward
        if prev_position is not None:
            prev_price = self._get_price(self.t-1, *prev_position)
            curr_price = self._get_price(self.t, *prev_position)
            reward += curr_price - prev_price
        
        if prev_position != self.position and prev_position is not None:
            reward -= self.transaction_cost
        self.cum_pnl +=reward
        
        self.t += 1
        # advance time
        if self.t >= len(self.dataframes):
            done = True
            next_obs = self._get_obs(last=True)
        else:
            next_obs = self._get_obs()
        return next_obs, reward, done, {}
    
    def _get_obs(self, last=False):
        t_idx = self.t
        if last:
            t_idx = self.t - 1
        df = self.dataframes[t_idx]
        S = df['underlying'].iloc[0]
        
        obs = np.zeros((self.n_strikes, self.n_types, 3), dtype=np.float32)
        for _, row in df.iterrows():
            i = self.strike_to_idx[row['strike']]
            ln_m = np.log(S / row['strike'])
            obs[i, 0, :] = [ln_m, row['call'], self.cum_pnl]
            obs[i, 1, :] = [ln_m, row['put'], self.cum_pnl]
        return obs
    
    def _get_price(self, t, strike_idx, type_idx):
        df = self.dataframes[t]
        strike = self.strikes[strike_idx]
        row = df[df['strike'] == strike]
        if type_idx == 0:
            return row['call'].values[0]
        else:
            return row['put'].values[0]

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)  # outputs probabilities over actions
        )

    def forward(self, x):
        return self.net(x)

class Brain:
  def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

  def elite_trajectory(self, experiences):
      total_rewards = [[i, sum([exp[2] for exp in single_experience])] for i, single_experience in enumerate(experiences)]
      best_idx = max(total_rewards, key=lambda x: x[1])[0]
      return experiences[best_idx]

  def compile(self, experience_universe, gamma=0.99):
    experiences = self.elite_trajectory(experience_universe)
    T = len(experiences)
    discounted_rewards = [0] * T
    G = 0
    for t in reversed(range(T)):
        _, _, reward, _, _ = experiences[t]
        G = reward + gamma * G
        discounted_rewards[t] = G
    training_data = []
    for i, exp in enumerate(experiences):
        state, action, _, _, _ = exp
        training_data.append((state, action, discounted_rewards[i]))
    return training_data

num_generations = 10
env = Wallstreet(input_data)
state_dim = env.n_strikes * env.n_types * 3
action_dim = env.action_space.n
agent = Brain(state_dim, action_dim)
exp_directory = []

for gen in range(num_generations):
    state = env.reset().astype(np.float32)
    done = False
    experiences = []

    while not done:
        state_tensor = torch.tensor(state.reshape(-1), dtype=torch.float32)
        
        if gen == 0:
            action = np.random.randint(env.action_space.n)
        else:
            with torch.no_grad():
                #print(state_tensor)
                action_probs = agent.policy_net(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
        
        next_state, reward, done, _ = env.step(action)
        experiences.append([state, action, reward, next_state, done])
        state = next_state

    exp_directory.append(experiences)
    training_data = agent.compile(exp_directory)
    Gs = np.array([G for _, _, G in training_data], dtype=np.float32)
    Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)
    training_data = [(state, action, Gs[i]) for i, (state, action, G) in enumerate(training_data)]
    for state, action, G in training_data:
        state_tensor = torch.tensor(state.reshape(-1), dtype=torch.float32)
        action_probs = agent.policy_net(state_tensor)
        log_prob = torch.log(action_probs[action])
        loss = -log_prob * G

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    avg_reward = sum([exp[2] for exp in experiences]) / len(experiences)
    #print(f"Generation {gen}, Average Reward: {avg_reward}")