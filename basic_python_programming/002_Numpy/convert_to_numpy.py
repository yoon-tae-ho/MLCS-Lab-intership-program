# %%
import os, glob
import numpy as np

path_to_outputs = os.path.join('..','001_CSV','outputs')

path_to_train_2d_datasets = os.path.join(path_to_outputs, 'train', '*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)

# We already know that the width and height of the 2D MNIST dataset is 28.
train = np.empty((len(train_2d_files), 28, 28))
train_labels = np.empty((len(train_2d_files)))

# Read the csv and replace the elements
for data_idx, data_path in enumerate(train_2d_files):
  csv_data = np.loadtxt(data_path, delimiter=",")
  train[data_idx] = csv_data
  label_idx = len(path_to_outputs + "/train/#")
  train_labels[data_idx] = data_path[label_idx]

# %%
# load test data
path_to_test_2d_datasets = os.path.join('..','001_CSV','outputs','test','*.csv')
test_2d_files = glob.glob(path_to_test_2d_datasets)

test = np.empty((len(test_2d_files), 28, 28))
test_labels = np.empty((len(test_2d_files)))

for data_idx, data_path in enumerate(test_2d_files):
  csv_data = np.loadtxt(data_path, delimiter=",")
  test[data_idx] = csv_data
  label_idx = len(path_to_outputs + "/test/#")
  test_labels[data_idx] = data_path[label_idx]

# %%
# concatenate train and test data
data = np.concatenate((train, test), axis=0)
labels = np.concatenate((train_labels, test_labels))

# %%
# shuffle
shuffled = np.array(list(range(len(labels))))
np.random.shuffle(shuffled)

# %%
# split
shuffled_len = len(shuffled)

# index_list == [train_indexes, validation_indexes, test_indexes]
index_list = np.split(shuffled, [int(shuffled_len / 10 * 7), int(shuffled_len / 10 * 9)])

# %%
# apply shuffled indexes

# 0: train
# 1: validation
# 2: test
data_list = []
label_list = []

for i, indexes in enumerate(index_list):
  a_data = np.empty((len(indexes), 28, 28))
  a_labels = np.empty(len(indexes))
  
  for j, index in enumerate(indexes):
    a_data[j] = data[index]
    a_labels[j] = labels[index]
  
  data_list.append(a_data)
  label_list.append(a_labels)

# %%
# output
path_to_save = os.path.join("dataset.npz")
np.savez(path_to_save, train_x=data_list[0], train_y=label_list[0], valid_x=data_list[1], valid_y=label_list[1], test_x=data_list[2], test_y=label_list[2])
