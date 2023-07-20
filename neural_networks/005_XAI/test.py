import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2


def preprocessing(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))

    return image, label


tfds.disable_progress_bar()

splits, info = tfds.load(
    "rock_paper_scissors", with_info=True, as_supervised=True, split=["train", "test"]
)
(train_images, test_images) = splits
num_examples = info.splits["train"].num_examples
num_classes = info.features["label"].num_classes
BATCH_SIZE = 32
print(info)
print(splits)
