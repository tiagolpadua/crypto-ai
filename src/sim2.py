import os
import datetime
import math
import urllib.request
from urllib.error import URLError, HTTPError
import zipfile
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM

# Constants
EPOCHS = 2
RANDOM_SEED = 42
DROPOUT = 0.2
SEQ_LEN = 100
WINDOW_SIZE = SEQ_LEN - 1
BATCH_SIZE = 64

# Functions
def download_data():
  pair = 'BTCUSDT'
  granularity = '5m'
  start_year = 2020
  end_year = 2023

  base_url = 'https://data.binance.vision/data/spot/monthly/klines'

  for year in range(start_year, end_year + 1):
    for month in range(1, 12 + 1):
      monthf = "{:0>2d}".format(month)
      fname = f'{pair}-{granularity}-{year}-{monthf}.zip'
      url = f'{base_url}/{pair}/{granularity}/{fname}'

      dpath = f"downloads/{fname}"

      if os.path.exists(dpath):
        print(f"{fname} already exists")

      else:
        print(f"Downloading {url}")
        try:
          urllib.request.urlretrieve(url, dpath)
        except HTTPError as e:
          # do something
          print('HTTPError: ', e.code)
        except URLError as e:
          # do something
          print('URLError: ', e.reason)

      if os.path.exists(dpath):
        with zipfile.ZipFile(dpath, 'r') as zip_ref:
          zip_ref.extractall('temp')

def create_results_dir():
    isotime = datetime.datetime.now().replace(microsecond=0).isoformat()
    parent_dir = "output"
    output_path = os.path.join(parent_dir, isotime)
    os.mkdir(output_path)
    return output_path

def hline(title = ""):
  size = 80
  print()
  if len(title) > 0:
    l = math.floor((size - len(title)) / 2) - 1
    print(("-" * l) + " " + title + " " + ("-" * l))
  else:
    print("-" * size)

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

utils.clear_temp_dir()

download_data()

exit()

np.random.seed(RANDOM_SEED)

results_path = create_results_dir()

# Data comes from:
# https://finance.yahoo.com/quote/BTC-USD/history?period1=1279314000&period2=1556053200&interval=1d&filter=history&frequency=1d

csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
# csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/AAPL.csv"

df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date')

hline('df.head()')
print(df.head())
# hline()

hline('df.shape')
print(df.shape)
# hline()

ax = df.plot(x='Date', y='Close')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
plt.savefig(os.path.join(results_path, '01-plot.png'))
plt.close()

scaler = MinMaxScaler()

close_price = df.Close.values.reshape(-1, 1)

scaled_close = scaler.fit_transform(close_price)

hline('scaled_close.shape')
print(scaled_close.shape)

hline('np.isnan(scaled_close).any()')
print(np.isnan(scaled_close).any())

scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1)

hline('np.isnan(scaled_close).any()')
print(np.isnan(scaled_close).any())

X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)

hline('X_train.shape')
print(X_train.shape)

hline('X_test.shape')
print(X_test.shape)

hline('Training...')

model = keras.Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('linear'))

model.compile(
    loss='mean_squared_error', 
    optimizer='adam'
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(results_path, '02-loss.png'))
plt.close()

hline('Predicting...')

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.savefig(os.path.join(results_path, '03-prediction.png'))
plt.close()

hline('THE END!')
