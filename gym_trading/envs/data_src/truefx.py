#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


class TrueFXDataSrc(object):

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
                                 names=['Symbol', 'datetime', 'Bid', 'Ask'],
                                 index_col=1, parse_dates=True
                                 )
                ohlc_df = df['Ask'].resample('15Min').ohlc()
                print("Saving the Dataframe to file")
                ohlc_df.to_csv(MIN_CSV_FILE_PATH)
            else:
                print("Loading the series from the file")
                series = pd.read_csv(
                    MIN_CSV_FILE_PATH, parse_dates=True, index_col=0,
                    usecols=['datetime', 'close'], squeeze=True
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
