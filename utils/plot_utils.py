import numpy as np
import matplotlib.pyplot as plt
import imageio
from functools import reduce
import math
from pathlib import Path
import datetime
import os


class PlotFactory():
    def __init__(self, prior_factory, results_dir, prior_type, n_classes=10, x_sampling_reconstr=17,
                 y_sampling_reconstr=17):

        super(PlotFactory, self).__init__()
        self.results_dir = results_dir
        self.prior_type = prior_type
        self.n_classes = n_classes
        self.x_sampling_reconstr = x_sampling_reconstr
        self.y_sampling_reconstr = y_sampling_reconstr

        self.results_dir_current = os.path.join(self.results_dir, (prior_type + '_' +
                                                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        Path(self.results_dir_current).mkdir(parents=True, exist_ok=True)

        self.prior_factory = prior_factory

    @staticmethod
    def merge_images(images):
        side_len = int(math.sqrt(len(images)))
        img_enum = [(i//side_len, x) for i, x in enumerate(images)]

        rows = tuple(reduce((lambda x, y: np.concatenate((x, y), axis=1)),
                            tuple(x[1] for x in img_enum if x[0] == row_no))
                     for row_no in range(side_len))

        img_arr = reduce((lambda x, y: np.concatenate((x, y), axis=0)), tuple(x for x in rows))

        return img_arr

    def save_images(self, images, name='reconstruction.jpg'):
        images = images.reshape(10*10, 28, 28)
        img_write = PlotFactory.merge_images(images)
        img_write = img_write * 255
        img_write = img_write.astype(np.uint8)
        imageio.imwrite(self.results_dir_current + "/"+name, img_write)

    def save_images_sampled(self, images, name='result.jpg'):
        images = images.reshape(self.x_sampling_reconstr*self.y_sampling_reconstr, 28, 28)
        img_write = PlotFactory.merge_images(images)
        img_write = (img_write * 255).astype(np.uint8)
        imageio.imwrite(self.results_dir_current + "/"+name, img_write)

    def plot_distribution(self, z, labels, name='z_distribution.png'):
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(labels, 1), marker='o', edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', self.n_classes))
        plt.colorbar(ticks=range(self.n_classes))
        axes = plt.gca()
        x_min, y_min = tuple(np.min(z, axis=0))
        x_max, y_max = tuple(np.max(z, axis=0))
        x_span, y_span = (x_max - x_min, y_max - y_min)
        axes.set_xlim([x_min - 0.2 * x_span, x_max + 0.2 * x_span])
        axes.set_ylim([y_min - 0.2 * y_span, y_max + 0.2 * y_span])
        plt.grid(True)

        plt.savefig(self.results_dir_current + "/" + name)
        plt.close()

    def plot_sampling_reconstr(self):
        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        # z_sample = np.rollaxis(np.mgrid[3.0:-3.0:15 * 1j, 3.0:-3.0:15 * 1j], 0, 3) #oryginal version
        x_range = self.prior_factory.gaussian_mixture_x_stddev * 3.0 * 2
        z_sample = np.rollaxis(np.mgrid[x_range:-x_range:self.x_sampling_reconstr * 1j, x_range:-x_range:self.y_sampling_reconstr * 1j], 0, 3)
        return z_sample.reshape([-1, 2])

    def plot_distribution_demo(self, prior_type, batch_size, n_classes=10, name=None):
        labels = np.random.randint(0, n_classes, size=[batch_size])
        z = self.prior_factory.get_prior(prior_type)(batch_size, labels, n_classes)
        if name is None:
            name = prior_type + "_target_prior.png"
        labels_one_hot = np.eye(len(labels))[labels]
        self.plot_distribution(z, labels_one_hot, name)
