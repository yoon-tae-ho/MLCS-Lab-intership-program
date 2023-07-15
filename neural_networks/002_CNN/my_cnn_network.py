import numpy as np

from keras.datasets import cifar10
from keras.applications import DenseNet121
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation

from sklearn.model_selection import train_test_split


class ImageClassification:
    def __init__(self) -> None:
        # constants
        self.test_ratio = 0.15
        self.valid_ratio = 0.15
        self.input_dim = (32, 32, 3)
        self.num_classes = 10
        self.trainable = False

    def normalize(self, x):
        x = x - x.min()
        x = x / x.max()
        return x

    def create(self):
        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # concatenate data
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # preprocessing
        x = x.astype("float32")
        x = self.normalize(x)
        y = to_categorical(y, 10)

        # split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_ratio, random_state=111
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=(self.valid_ratio / (1 - self.test_ratio)),
            random_state=111,
        )

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def build(self):
        # load base model
        base_model = DenseNet121(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_dim,
            classes=self.num_classes,
            classifier_activation="softmax",
        )
        base_model.trainable = self.trainable

        # add layers to base model
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        model.compile(
            loss="MSE",
            optimizer="adam",
            metrics=["accuracy"],
        )

        return model
