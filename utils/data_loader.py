# coding=utf-8
"""Loading data from tensorflow_datasets, creating train and test datasets."""


import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader():
    def __init__(self, batch_size):
        super(DataLoader, self).__init__()

        self.batch_size = batch_size

        self.mnist_data, self.info = tfds.load(name='mnist', shuffle_files=True, with_info=True)
        self.img_shape = self.info.features['image'].shape
        self.img_size_x = self.img_shape[0]
        self.img_size_y = self.img_shape[1]
        self.n_classes = self.info.features['label'].num_classes

        self.train_ds = None
        self.test_ds = None

    def make_dataset(self):
        """Constructs training and test datasets.

        Returns:
            Tuple (tf.data.Dataset, tf.data.Dataset):
                Tuple containing training (first) and test (second) datasets.

        """

        train_ds, test_ds = self.mnist_data['train'], self.mnist_data['test']
        train_ds = train_ds.map(lambda ds: {'image': tf.cast(ds['image'], tf.float32) / 255.,
                                            'label': ds['label']})
        train_ds = train_ds.map(lambda ds:
                                (tf.reshape(ds['image'], (self.img_size_x * self.img_size_y,)),
                                 tf.one_hot(ds['label'], self.n_classes)))
        train_ds.shuffle(50000)
        train_ds = train_ds.batch(self.batch_size)

        assert isinstance(train_ds, tf.data.Dataset)

        test_ds = test_ds.map(lambda ds: {'image': tf.cast(ds['image'], tf.float32) / 255.,
                                          'label': ds['label']})
        test_ds = test_ds.map(lambda ds:
                              (tf.reshape(ds['image'], (self.img_size_x * self.img_size_y,)),
                               tf.one_hot(ds['label'], self.n_classes)))
        self.train_ds, self.test_ds = train_ds, test_ds
        return self.train_ds, self.test_ds

    def get_test_sample(self, num_samples, batch_size):
        """Returns a sample from the test dataset as a tuple"""
        ret_ds = self.test_ds.take(num_samples).batch(batch_size)
        return tuple(tfds.as_numpy(ret_ds))[0]
