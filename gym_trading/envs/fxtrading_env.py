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
        # self.series = np.vstack([np.array([data[key][i-window:i]for i in range(window, (len(data[key]) + 1))]) for key in list(data.keys())])

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



class FXData(object):

    def __init__(self):
        self.symbol = 'EURUSD'

        print("Loading the files from the system")
        df = pd.read_csv("/Users/sushantkumar/Github/gym-trading/fx_data/{}-2017-01.csv".format(self.symbol),
            names=['Symbol', 'Datetime', 'Bid', 'Ask'], index_col=1, parse_dates=True)

        series = df['Ask'].resample('15Min').ohlc()['close']

        self.data = series.values
        self.n = len(self.data)


class FXTradingEnv(gym.Env):
    metadata = { 'render.modes': ['human'] }

    def __init__(self, window=8):
        self.src = FXData()
        self.data = self.src.data
        self.window = window
        self.n = 96

        self.action_space = spaces.Discrete(3)
        self.observation_space = Series(self.src.data, self.window)

        self.reset()


    def step(self, action):
        assert self.action_space.contains(action)

        if self.done:
            raise ValueError("Please do not call step once the env is done!")


        print("Action: {}".format(action))
        self.index += 1
        obs = self.data[(self.index - self.window):self.index]
        reward = (action - 1) * (self.data[(self.index - 1)] - self.data[(self.index - 2)])
        done = False
        info = {}


        print("Index and n: {} and {}".format(self.index, self.n))
        if (self.index) >= self.n:
            self.done = True

        return obs, reward, done, info


    def reset(self, test=False):
        self.index = self.window
        obs = self.data[:self.index]
        self.done = False
        return obs
