#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Simulate trading environment"

import logging.config
import math
import pkg_resources
import random

import cfg_load
import pandas

import numpy as np


import gym
from gym import spaces

import pandas as pd
import requests
import os
import calendar
import zipfile

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

DATA_FOLDER = "fx_data"

# TrueFX has the following Currency Pairs
# AUDJPY, AUDNZD, AUDUSD, CADJPY, CHFJPY, EURCHF, EURGBP, EURJPY,
# EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY

class Series(gym.Space):

    def __init__(self, data, window):
        self.shape = (window,)

        # For multiple stocks
        # self.series = np.vstack([np.array([data[key][i-window:i]for i in range(window, (len(data[key]) + 1))]) for key in list(data.keys())])

    def sample(self):
        return self.series[np.random.randint(len(self.series))]


class FXData(object):

    def __init__(self):
        self.symbol = 'EURUSD'
        self.year = '2017'

        self.download_data()


        self.series = self.preprocess_data()
        self.data = self.series.values
        self.n = len(self.data)


    # Loads the data from the system to the memory
    def preprocess_data(self):


        print("Pre-processing the data")
        re_series = pd.Series([])
        for month in range(1, 3):

            MIN_CSV_FILE_NAME = '{}-{}-{:02}-15min.csv'.format(self.symbol, self.year, month)
            MIN_CSV_FILE_PATH = '{}/{}/{:02}/{}'.format(
                DATA_FOLDER, self.year, month, MIN_CSV_FILE_NAME
            )

            # Checks if the pre-processed file already exists
            # If not, then creates one and saves it for faster turnaround time.
            if not os.path.isfile(MIN_CSV_FILE_PATH):
                FILE_NAME = '{}-{}-{:02}.zip'.format(self.symbol, self.year, month)
                FILE_PATH = '{}/{}/{:02}/{}'.format(
                    DATA_FOLDER, self.year, month, FILE_NAME
                )

                print("Processing the file: {}".format(FILE_PATH))
                zf = zipfile.ZipFile(FILE_PATH)
                CSV_FILE = zf.namelist()[0]
                df = pd.read_csv(zf.open(CSV_FILE),
                                 names = ['Symbol', 'datetime', 'Bid', 'Ask'], index_col = 1,
                                 parse_dates=True
                                 )
                ohlc_df = df['Ask'].resample('15Min').ohlc()
                print("Saving the Dataframe to file")
                ohlc_df.to_csv(MIN_CSV_FILE_PATH)
            else:
                print("Loading the series from the file")
                series = pd.read_csv(
                    MIN_CSV_FILE_PATH, parse_dates=True, index_col=0, usecols=['datetime', 'close'],
                    squeeze=True
                )

                re_series = pd.concat([re_series, series])

        return re_series


    # Downloads the files, if they are not already downloaded
    def download_data(self):
        # Downloading the data
        for month in range(1, 3):
            FILE_NAME = '{}-{}-{:02}.zip'.format(self.symbol, self.year, month)
            month_name = calendar.month_name[month].upper()
            ZIP_FILE_URL = 'https://www.truefx.com/dev/data/{}/{}-{}/{}'.format(
                self.year, month_name, self.year, FILE_NAME)

            DOWNLOAD_FILE_PATH = '{}/{}/{:02}/{}'.format(
                DATA_FOLDER, self.year, month, FILE_NAME
            )

            # Check if file already exists
            if os.path.isfile(DOWNLOAD_FILE_PATH):
                print("{} file already exists".format(DOWNLOAD_FILE_PATH))
                continue

            # Check if the folders exists, if not makedirs
            if not os.path.exists(os.path.dirname(DOWNLOAD_FILE_PATH)):
                os.makedirs(os.path.dirname(DOWNLOAD_FILE_PATH))

            # Download the file and save it in the right path
            print("Downloading {}".format(ZIP_FILE_URL))
            r = requests.get(ZIP_FILE_URL, verify=False)
            with open(DOWNLOAD_FILE_PATH, 'wb') as f:
                f.write(r.content)



class Tradenv(gym.Env):
    """
    Define a simple trading environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("Tradenv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.src = FXData()
        self.data = self.src.data
        self.series = self.src.series
        self.n = len(self.series)


        self.done = FALSE
        self.index = 0

        # Define what the agent can do
        self.action_space = spaces.Discrete(3)



        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

    past = 0
    v = 0
    def step(self, action), sp:
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        assert self.action_space.contains(action)
        if self.done:
            raise ValueError("Please do not call step once the env is done!")
        if self.index >= self.n :
            self.done = True

        self.index += 1
        print("Action: {}".format(action - 1))
    # noinspection PyGlobalUndefined
    global past_v = v
        v = v + action *  (self.series[self.index] - self.series[self.index - 1]) + sp * abs(action - past)
        past = action
        reward = self._get_reward()
        ob = self._get_observation()
        return ob, reward, self.done, {}


    def _get_reward(self):
        """Reward is each trade."""
        reward = log(self.v/past_v)
        return(reward)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.index = 0
        self.done = False
        return self.get_obs()

    def _render(self, mode='human', close=False):
        return

    def _get_observation(self):
        """Get the observation."""
        stock_price = self.series[self.index]
        time_stamp = self.series.index[self.index]
        ob = [stock_price ,time_stamp ]
        return ob


