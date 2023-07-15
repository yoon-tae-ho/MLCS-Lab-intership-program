import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.initializers import glorot_normal
from keras.optimizers import RMSprop, SGD, Adam
from keras.constraints import unit_norm
from keras import regularizers
from keras.engine import Model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from cnn_network import ImageClassification

batch_size=256
epochs=15
epochs_2=12
learn_rate=0.001
input_dim=(32, 32, 3)
output_dim=10
test_size=0.1
valid_size=0.1/0.9
weight_decay = 1e-4

IC=ImageClassification()
x_train, x_valid, x_test, y_train, y_valid, y_test=IC.create()
model=IC.build()
history=model.fit(x_train, y_train,
                    epochs=epochs_2,
                    batch_size=batch_size,
                    validation_data=(x_valid,y_valid),
                    verbose=1)


score=model.evaluate(x_test, y_test, verbose=0)



print('Test loss for model:', score[0])
print('Test accuracy for model:', score[1])
    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
