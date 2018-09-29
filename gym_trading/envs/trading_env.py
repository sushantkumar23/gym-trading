# trading_env.py

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import requests

import pandas as pd
import numpy as np


class Series(gym.Space):

    def __init__(self, data, window):
        self.shape = (window,)

        # For multiple stocks
        self.series = np.vstack([np.array([data[key][i-window:i]for i in range(window, (len(data[key]) + 1))]) for key in list(data.keys())])

    def sample(self):
        return self.series[np.random.randint(len(self.series))]

class USEquityDailyDataSource:

    def __init__(self):
        self.symbols = ['MSFT', 'IBM', 'QCOM']
        self.data = {}
        self.n = {}
        for symbol in self.symbols:
            temp_series = pd.read_csv('/Users/sushantkumar/Github/gym-trading/data/daily_{}.csv'.format(symbol), usecols=['close'], squeeze=True)
            temp_series = temp_series.sort_index()
            self.data[symbol] = temp_series.values
            self.n[symbol] = len(self.data[symbol])


class TradingEnv(gym.Env):
    metadata = { 'render.modes': ['human'] }

    def __init__(self):
        self.src = USEquityDailyDataSource()
        self.src_data = self.src.data[self.src.symbols[0]]
        self.window = 5
        self.n = 100

        self.test = False
        self.test_split = 0.2

        self.test_index = int(len(self.src_data)*(1-self.test_split))
        self.train_data = self.src_data[:self.test_index]
        self.test_data = self.src_data[self.test_index:]

        self.action_space = spaces.Discrete(3)
        self.observation_space = Series(self.src.data, self.window)

        # Set Debug mode
        self.debug_mode = False

        self.reset()


    def step(self, action):
        assert self.action_space.contains(action)

        trades = np.abs(self.action - action)

        self.index += 1
        obs = self.data[(self.index - self.window):self.index]
        reward = (action - 1) * (self.data[(self.index - 1)] - self.data[(self.index - 2)])
        done = False
        info = {}

        self.action = action
        if (self.index) >= self.n:
            done = True

        return obs, reward, done, info


    def reset(self, test=False):
        self.test = test
        self._set_data()
        self.trades = 0
        self.action = 0
        self.index = self.window
        obs = self.data[:self.index]
        return obs


    def render(self, mode='human', close=False):
        pass


    def _set_data(self):
        if self.test:
            start = np.random.randint((len(self.test_data) - self.n) + 1)
            self.data = self.test_data[start:(start+self.n)]
        else:
            start = np.random.randint((len(self.train_data) - self.n) + 1)
            self.data = self.train_data[start:(start+self.n)]

        if self.debug_mode:
            print("Start Index: {}".format(start))
