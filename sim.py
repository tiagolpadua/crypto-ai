import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
import datetime
# from tensorflow.python.keras.layers import CuDNNLSTM
# from tensorflow.keras.models import Sequential

# %matplotlib inline

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 14, 8

isotime = datetime.datetime.now().replace(microsecond=0).isoformat()

print(isotime)

parent_dir = "output"

output_path = os.path.join(parent_dir, isotime)

os.mkdir(output_path)

# exit()

def hline(title = ""):
  size = 80

  if len(title) > 0:
    l = math.floor((size - len(title)) / 2) - 1
    print(("-" * l) + " " + title + " " + ("-" * l))
  else:
    print("-" * size)

RANDOM_SEED = 42

EPOCHS = 40

np.random.seed(RANDOM_SEED)

# Data comes from:
# https://finance.yahoo.com/quote/BTC-USD/history?period1=1279314000&period2=1556053200&interval=1d&filter=history&frequency=1d

csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/BTC-USD.csv"
# csv_path = "https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/3.stock-prediction/AAPL.csv"

df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date')

hline('df.head()')
print(df.head())
hline()

hline('df.shape')
print(df.shape)
hline()

ax = df.plot(x='Date', y='Close')
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
plt.savefig(os.path.join(output_path, 'plot.png'))
plt.close()

scaler = MinMaxScaler()

close_price = df.Close.values.reshape(-1, 1)

scaled_close = scaler.fit_transform(close_price)

hline('scaled_close.shape')
print(scaled_close.shape)
# print(scaled_close)
hline()

hline('np.isnan(scaled_close).any()')
print(np.isnan(scaled_close).any())
hline()

scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1)

hline('np.isnan(scaled_close).any()')
# print(scaled_close)
print(np.isnan(scaled_close).any())
hline()

SEQ_LEN = 100

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


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)

hline('X_train.shape')
print(X_train.shape)
hline()

hline('X_test.shape')
print(X_test.shape)
hline()

DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

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

BATCH_SIZE = 64

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
plt.savefig(os.path.join(output_path, 'loss.png'))
plt.close()

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.savefig(os.path.join(output_path, 'prediction.png'))
plt.close()

hline()
hline('the end!')
hline()