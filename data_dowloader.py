import os
import datetime
import urllib.request
import zipfile
from urllib.error import URLError, HTTPError

def download_data(pair = 'BTCUSDT', granularity = '5m', start_month = 1, start_year = 2020):

  today = datetime.date.today()

  current_year = today.year

  current_month = today.month

  BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines'

  DOWNLOAD_DIR = "downloads"

  TEMP_DIR = "temp"

  for year in range(start_year, current_year + 1):
    for month in range(1, 12 + 1):

      # ignore future year/months
      if (year == current_year and month > current_month):
        continue

      # ignore months before start month
      if (year == start_year and month < start_month):
        continue

      monthf = "{:0>2d}".format(month)
      fname = f'{pair}-{granularity}-{year}-{monthf}.zip'
      url = f'{BASE_URL}/{pair}/{granularity}/{fname}'

      dpath = os.path.join(DOWNLOAD_DIR, fname)

      if os.path.exists(dpath):
        print(f"{fname} already exists")

      else:
        print(f"Downloading {url}...")
        try:
          urllib.request.urlretrieve(url, dpath)
        except HTTPError as e:
          print('HTTPError: ', e.code)
        except URLError as e:
          print('URLError: ', e.reason)

      if os.path.exists(dpath):
        with zipfile.ZipFile(dpath, 'r') as zip_ref:
          zip_ref.extractall(TEMP_DIR)