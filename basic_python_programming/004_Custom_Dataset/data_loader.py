import os, glob
import csv
import numpy as np
import random


class DataLoader:
    def __init__(
        self, dataset_name="starbucks_beverage", test=False, shuffle=True
    ) -> None:
        self.categories = []  # category of beverage list
        self.beverage_features = []  # features of beverage except category
        self.dataset_name = dataset_name
        self.test = test
        self.shuffle = shuffle

    def create(self) -> None:
        csvs = sorted(glob.glob(os.path.join("datasets", "*.csv")))

        if self.shuffle:
            random.shuffle(csvs)

        for _csv in csvs:
            with open(_csv, "r") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=",")
                next(csv_reader, None)  # skip the headers
                for row in csv_reader:
                    self.categories.append(row[0])
                    self.beverage_features.append(row[1:])

        np.savez(
            os.path.join("datasets", self.dataset_name + ".npz"),
            categories=self.categories,
            beverage_features=self.beverage_features,
        )

    def load(self, ratio1=0.7, ratio2=0.9):
        # train, validation, test set size ratio
        d = np.load(os.path.join("datasets", self.dataset_name + ".npz"))
        self.categories = d["categories"]
        self.beverage_features = d["beverage_features"]

        length = len(self.categories)
        (
            self.categories_train,
            self.categories_validation,
            self.categories_test,
        ) = np.split(self.categories, [int(length * ratio1), int(length * ratio2)])
        (
            self.beverage_features_train,
            self.beverage_features_validation,
            self.beverage_features_test,
        ) = np.split(
            self.beverage_features, [int(length * ratio1), int(length * ratio2)]
        )

        return {
            "categories_train": self.categories_train,
            "categories_validation": self.categories_validation,
            "categories_test": self.categories_test,
            "beverage_features_train": self.beverage_features_train,
            "beverage_features_validation": self.beverage_features_validation,
            "beverage_features_test": self.beverage_features_test,
        }


if __name__ == "__main__":
    dl = DataLoader()
    dl.create()
    result = dl.load()
