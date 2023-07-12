import numpy as np
import keras
from keras.optimizers import RMSprop

# load models and data
model_1 = keras.models.load_model("./datasets/model_1.h5")
model_2 = keras.models.load_model("./datasets/model_2.h5")
data = np.load("./datasets/test_data.npz")
x_test = data["x_test"]
y_test = data["y_test"]

# evaluate model
score_1 = model_1.evaluate(x_test, y_test, verbose=0)
print("Test loss 1:", score_1[0])
print("Test accuracy 1:", score_1[1])
score_2 = model_2.evaluate(x_test, y_test, verbose=0)
print("Test loss 2:", score_2[0])
print("Test accuracy 2:", score_2[1])
