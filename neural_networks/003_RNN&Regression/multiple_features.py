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
num_of_features = 6
input_dim = (days, num_of_features)
close_price_index = 3


# load data
yf.pdr_override()
df = pdr.get_data_yahoo("SBUX", start="2018-01-01", end="2023-07-15")
values = df.values

# data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

x = []
y = []

for i in range(days, len(scaled_data)):
    x.append(scaled_data[i - days : i])
    y.append(scaled_data[i])

x, y = np.array(x), np.array(y)

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
model.add(Dense(num_of_features))

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
predictions = scaler.inverse_transform(predictions)[:, close_price_index]
y_test = scaler.inverse_transform(y_test)[:, close_price_index]

rmse = np.sqrt(np.mean((predictions - y_test) ** 2, axis=0))
print("rmse: ", rmse, "$")

# for plotting
GT = pd.DataFrame(df[(days + x_train.shape[0] + x_valid.shape[0]) :])
GT["Predictions"] = predictions

plt.figure()
plt.title("Result")
plt.xlabel("Date")
plt.ylabel("Close prise USD ($)")
plt.plot(GT[["Close", "Predictions"]])
plt.legend(["Ground Truth", "Predictions"], loc="lower right")
plt.show()
