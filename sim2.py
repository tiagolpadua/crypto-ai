import os
import glob
from utils import clear_temp_dir, hline, create_output_dir
from data_dowloader import download_data

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
TEMP_DIR = "temp"

# Functions
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

clear_temp_dir()

download_data()

np.random.seed(RANDOM_SEED)

results_path = create_output_dir()

csv_files = glob.glob(os.path.join(TEMP_DIR, "*.csv"))

dfs = []
for csv_file in csv_files:
  # Read each CSV file into DataFrame
  # This creates a list of dataframes
  df = pd.read_csv(csv_file, index_col=None, header=None, names=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "unused"])

  df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='ms')
  df['close_time_dt'] = pd.to_datetime(df['close_time'], unit='ms')
  df['close_time_rd'] = df['close_time'] + 1
  df['close_time_rd_dt'] = pd.to_datetime(df['close_time_rd'], unit='ms')

  dfs.append(df)

df = pd.concat(dfs)

df = df.sort_values(by=['open_time'])

hline('df')
print(df)

hline('df.shape')
print(df.shape)

ax = df.plot(x='close_time_rd_dt', y='close')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
plt.savefig(os.path.join(results_path, '01-plot.png'))
plt.close()

scaler = MinMaxScaler()

close_price = df.close.values.reshape(-1, 1)

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

output_df = pd.DataFrame({'y_test_inverse': y_test_inverse[0:, 0], 'y_hat_inverse': y_hat_inverse[0:, 0]})

output_df.to_excel(os.path.join(results_path, '04-prediction.xlsx'), index=False)

output_df.to_csv(os.path.join(results_path, '05-prediction.csv'), index=False)

hline('THE END!')
