# truefx_db.py

import psycopg2
import pandas as pd
import os

URL = "postgresql://postgres:wciCbjFk9oKHBka6@35.200.209.205/truefx"


class TrueFXData(object):

    def __init__(self,
                 symbol='EURUSD',
                 start_date='2012-01-01',
                 end_date='2012-03-31'):
        """Initialises the Database instance for the TrueFX data which is
        currently uploaded on a database on the GCP.

        Parameters:
            symbol (string): currency pair for which the data needs to be
                downloaded
            start_date (date_string): start date from which the currency pair
                candles should be retrieved
            end_date (date_string): end date to which the currency pair candles
                should be retrieved.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data()

    def get_data(self):
        """Connects to the database downloads the data and returns the candles
        as dataframe object"""

        print("Connecting to the Postgres on Google Cloud")
        DATABASE_URL = os.environ.get('DATABASE_URL', URL)
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM EURUSD \
            WHERE datetime >= (%s) \
            AND datetime < (%s);""",
            (self.start_date, self.end_date))
        results = cur.fetchall()
        cur.close()
        conn.close()

        df = pd.DataFrame.from_records(
            results,
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            index='datetime',
            coerce_float=True)

        return df
