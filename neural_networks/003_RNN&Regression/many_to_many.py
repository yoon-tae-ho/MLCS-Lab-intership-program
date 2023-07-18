import yfinance as yf
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# constants
train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
days = 60  # length of a data
bs = 30  # batch_size
epoch = 20  # epoch
time_steps = 3
input_dim = (days, 1)


# load data
yf.pdr_override()
df = pdr.get_data_yahoo("SBUX", start="2018-01-01", end="2023-07-15")
values = df["Close"].values

# # data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))

x = []
y = []

for i in range(days, len(scaled_data) - time_steps):
    x.append(scaled_data[i - days : i])
    y.append(scaled_data[i : i + time_steps])

x = np.array(x)
y = np.array(y)

data_len = y.shape[0]
x_train, x_valid, x_test = np.split(
    x, [int(data_len * train_ratio), int(data_len * (train_ratio + valid_ratio))]
)
y_train, y_valid, y_test = np.split(
    y, [int(data_len * train_ratio), int(data_len * (train_ratio + valid_ratio))]
)


# set model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=input_dim))
model.add(Dense(256))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(64))
model.add(Dense(3))

model.summary()
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(
    x_train,
    y_train,
    batch_size=bs,
    epochs=epoch,
    validation_data=(x_valid, y_valid),
    verbose=1,
)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(y_test.shape[0], y_test.shape[1]))
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("rmse: ", rmse, "$")
