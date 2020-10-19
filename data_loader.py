import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from google.cloud import storage
from io import StringIO
from tensorflow.python.lib.io import file_io
import random


class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

        self.train = None
        self.test = None

    def _load_data(self):
        # Please make sure to change this function to load your train/validation/test data.
        train_data = np.array([plt.imread('./data/test_images/0.jpg'), plt.imread('./data/test_images/1.jpg'),
                      plt.imread('./data/test_images/2.jpg'), plt.imread('./data/test_images/3.jpg')])
        self.X_train = train_data
        self.y_train = np.array([284, 264, 682, 2])

        print(self.X_train.shape)
        print(self.y_train.shape)

        val_data = np.array([plt.imread('./data/test_images/0.jpg'), plt.imread('./data/test_images/1.jpg'),
                    plt.imread('./data/test_images/2.jpg'), plt.imread('./data/test_images/3.jpg')])

        self.X_val = val_data
        self.y_val = np.array([284, 264, 682, 2])

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 224
        img_width = 224
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def _load_data(self):
        x = np.load(StringIO(file_io.read_file_to_string('gs://tinynet/dataset/cifar10/X_train.npy')))
        y = np.load(StringIO(file_io.read_file_to_string('gs://tinynet/dataset/cifar10/y_train.npy')))
        ex = np.load(StringIO(file_io.read_file_to_string('gs://tinynet/dataset/cifar10/X_val.npy')))
        ey = np.load(StringIO(file_io.read_file_to_string('gs://tinynet/dataset/cifar10/y_val.npy')))

        self.X_train = x
        self.y_train = y

        self.X_val = ex
        self.y_val = ey

        self.X_test = ex
        self.y_test = ey

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 224
        img_width = 224
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def load_data(self):
        if os.path.exists("./data/X_train.npy"):
            self.X_train = np.load("./data/X_train.npy")
            self.X_test = np.load("./data/X_val.npy")
            self.X_val = self.X_test

            self.y_train = np.load("./data/y_train.npy")
            self.y_test = np.load("./data/y_val.npy")
            self.y_val = self.y_test
        else:
            dir = os.getcwd()

            os.chdir("./data/road/processed")

            data = []
            label = []

            f_label_names = os.listdir(".")
            try:
                f_label_names.remove(".DS_Store")
            except ValueError:
                pass
            try:
                f_label_names.remove("true")
            except ValueError:
                pass

            print("f_label start")

            for fname in f_label_names:
                img = cv2.imread(fname)
                data.append(img)
                label.append(0)

            print("f_label done")

            os.chdir("./true")

            t_label_names = os.listdir(".")
            try:
                t_label_names.remove(".DS_Store")
            except ValueError:
                pass

            print("t_label start")

            for fname in t_label_names:
                img = cv2.imread(fname)
                index = random.randint(0, len(data) - 1)
                data.insert(index, img)
                label.insert(index, 1)

            print("t_label done")

            data = np.array(data)
            label = np.array(label)
            print(data.shape)

            os.chdir(dir)

            cut = int(len(data) * 0.8)
            self.X_train = data[:cut]
            self.y_train = label[:cut]
            self.X_test = data[cut:]
            self.y_test = label[cut:]
            self.X_val = data[cut:]
            self.y_val = label[cut:]

            np.save("./data/X_train", self.X_train)
            np.save("./data/y_train", self.y_train)
            np.save("./data/X_val", self.X_val)
            np.save("./data/y_val", self.y_val)

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 224
        img_width = 224
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def _load_data(self):
        print("Downloading Datasets...")
        (x, y), (ex, ey) = datasets.cifar10.load_data()

        # array flattening
        y = y.reshape(y.shape[0])
        ey = ey.reshape(ey.shape[0])

        self.X_train = x
        self.y_train = y
        self.X_val = ex
        self.y_val = ey
        self.X_test = ex
        self.y_test = ey

        self.train_data_len = x.shape[0]
        self.val_data_len = ex.shape[0]
        img_height = 224
        img_width = 224
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def _resize(self, X):
        return cv2.resize(X, (224, 224))

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                X_batch = np.array(list(map(self._resize, X_batch)))
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                X_batch = np.array(list(map(self._resize, X_batch)))
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                X_batch = np.array(list(map(self._resize, X_batch)))
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/uitaekim/Desktop/workspace/HazardObjectDetection/TinyNet-d01b0f17b10a.json"
    dl = DataLoader(32)
    dl.load_data()

    print(dl.X_train.shape)
    print(dl.y_train.shape)
    print(len(dl.y_train) - np.count_nonzero(dl.y_train))
    print(np.count_nonzero(dl.y_train))
    print(dl.X_test.shape)
    print(dl.y_test.shape)
    print(len(dl.y_test) - np.count_nonzero(dl.y_test))
    print(np.count_nonzero(dl.y_test))
