import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# parameters
num_classes = 10

batch_size_1 = 128
epochs_1 = 20

batch_size_2 = 64
epochs_2 = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# concatenate train and test data
x_concat = np.concatenate((x_train, x_test), axis=0)
y_concat = np.concatenate((y_train, y_test), axis=0)

# preprocessing
x_concat = np.reshape(x_concat, (70000, 784))
x_concat = x_concat.astype("float32")
x_concat /= 255
y_concat = keras.utils.to_categorical(y_concat, num_classes)

# split
# train : validation : test = 7.0 : 1.5 : 1.5
length = len(y_concat)
x_train, x_validation, x_test = np.split(
    x_concat, [int(length * 0.7), int(length * 0.85)]
)
y_train, y_validation, y_test = np.split(
    y_concat, [int(length * 0.7), int(length * 0.85)]
)

# design model
# model 1
model_1 = Sequential()
model_1.add(Dense(512, activation="relu", input_shape=(784,)))
model_1.add(Dropout(0.2))
model_1.add(Dense(512, activation="relu"))
model_1.add(Dropout(0.2))
model_1.add(Dense(num_classes, activation="softmax"))

# model 2
model_2 = Sequential()
model_2.add(Dense(258, activation="relu", input_shape=(784,)))
model_2.add(Dropout(0.2))
model_2.add(Dense(258, activation="relu"))
model_2.add(Dropout(0.2))
model_2.add(Dense(num_classes, activation="softmax"))

# compile model
model_1.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)
model_2.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)

# fit model
histor_1 = model_1.fit(
    x_train,
    y_train,
    batch_size=batch_size_1,
    epochs=epochs_1,
    verbose=1,
    validation_data=(x_validation, y_validation),
)
histor_2 = model_2.fit(
    x_train,
    y_train,
    batch_size=batch_size_2,
    epochs=epochs_2,
    verbose=1,
    validation_data=(x_validation, y_validation),
)

# save models and data
np.savez(os.path.join("datasets", "test_data.npz"), x_test=x_test, y_test=y_test)
model_1.save(os.path.join("datasets", "model_1.h5"))
model_2.save(os.path.join("datasets", "model_2.h5"))
