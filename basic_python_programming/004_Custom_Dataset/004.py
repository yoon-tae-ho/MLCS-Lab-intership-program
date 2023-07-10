# Classification of songs' genre with feature datas

import os, glob
import csv
import numpy as np
import random

# number of datas (csv file) in datasets folder
data_number = len(glob.glob(os.path.join("datasets", "*.csv")))

# ratio for splitting train, validation, test datas
ratio1 = (int)(data_number * 0.7)
ratio2 = (int)(data_number * 0.9)

train_number = int(data_number * ratio1)
valid_number = int(data_number * (ratio2 - ratio1))
test_number = int(data_number - train_number - valid_number)


class DataLoader:
    def __init__(self, dataset_name="songs_in_spotify", test=False, shuffle=True):
        self.x = []  # array for each song's various features (ex: danceability, ...)
        self.y = []  # array for each song's genre(label)
        self.dataset_name = dataset_name
        self.shuffle = shuffle

    def create(self):
        # 'Create' imports all the raw dataset
        # numerical datas of song's each feature are already normalized.
        csvs = sorted(glob.glob(os.path.join("datasets", "*.csv")))
        if self.shuffle:
            random.shuffle(csvs)

        for a in csvs:
            with open(a, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    self.x.append(row[:-1])
                    self.y.append(
                        row[-1]
                    )  # genre data is saved in row[-1] and encoded into numerical datas.

        np.savez(
            os.path.join("./datasets1", self.dataset_name + ".npz"), x=self.x, y=self.y
        )

    def load(self):
        # 12 means the number of each features
        # features: danceability, energy, key, loudness, mode, speechiness,
        # acousticness, instrumentalness, liveness, valence, tempo, duration_ms
        self.x_train = (train_number, 12)
        self.y_train = train_number
        self.x_valid = (valid_number, 12)
        self.y_valid = valid_number
        self.x_test = (test_number, 12)
        self.y_test = test_number

        d = np.load("./datasets1/" + self.dataset_name + ".npz")

        # split total datas into train and test data
        [self.x_train, self.x_valid, self.x_test] = np.split(d["x"], [ratio1, ratio2])
        [self.y_train, self.y_valid, self.y_test] = np.split(d["y"], [ratio1, ratio2])

        return (
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            self.x_test,
            self.y_test,
        )


if __name__ == "__main__":
    dl = DataLoader()
    dl.create()
    dl.load()
