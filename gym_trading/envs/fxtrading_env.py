# trading_env.py

import gym
from gym import spaces

import pandas as pd
import numpy as np
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
        for month in range(1, 13):

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
        for month in range(1, 13):
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


class FXTradingEnv(gym.Env):
    metadata = { 'render.modes': ['human'] }

    def __init__(self, window=8):
        self.src = FXData()
        self.data = self.src.data
        self.series = self.src.series
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
        obs = self.series[(self.index - self.window):self.index]
        reward = (action - 1) * (self.data[(self.index - 1)] - self.data[(self.index - 2)])
        done = False
        info = {}

        print("Index and n: {} and {}".format(self.index, self.n))
        if (self.index) >= self.n:
            self.done = True

        return obs, reward, done, info


    def reset(self):
        self.index = self.window
        obs = self.series[:self.index]
        self.done = False
        return obs
