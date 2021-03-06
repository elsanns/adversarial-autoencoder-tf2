# coding=utf-8
"""Generates samples from supported types of prior distributions."""

import numpy as np
from math import sin, cos, sqrt


class PriorFactory:
    """Class containing methods for generation of samples
    from supported prior distributions.
    """

    def __init__(self, n_classes, gm_x_stddev=0.5, gm_y_stddev=0.1):
        super(PriorFactory, self).__init__()
        self.n_classes = n_classes
        self.gaussian_mixture_x_stddev = gm_x_stddev
        self.gaussian_mixture_y_stddev = gm_y_stddev

    def gaussian_mixture(self, batch_size, labels, n_classes):
        x_stddev = self.gaussian_mixture_x_stddev
        y_stddev = self.gaussian_mixture_y_stddev
        shift = 3 * x_stddev

        x = np.random.normal(0, x_stddev, batch_size).astype("float32") + shift
        y = np.random.normal(0, y_stddev, batch_size).astype("float32")
        z = np.array([[xx, yy] for xx, yy in zip(x, y)])

        def rotate(z, label):
            angle = label * 2.0 * np.pi / n_classes
            rotation_matrix = np.array(
                [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
            )
            z[np.where(labels == label)] = np.array(
                [
                    rotation_matrix.dot(np.array(point))
                    for point in z[np.where(labels == label)]
                ]
            )
            return z

        for label in set(labels):
            rotate(z, label)

        return z

    # Borrowed from https://github.com/nicklhy/AdversarialAutoEncoder/blob/master/data_factory.py#L40 (modified)
    def swiss_roll(self, batch_size, labels, n_classes):
        def sample(label, n_labels):
            uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(
                n_labels
            )
            r = sqrt(uni) * 3.0
            rad = np.pi * 4.0 * sqrt(uni)
            x = r * cos(rad)
            y = r * sin(rad)
            return np.array([x, y]).reshape((2,))

        dim_z = 2
        z = np.zeros((batch_size, dim_z), dtype=np.float32)
        for batch in range(batch_size):
            z[batch, :] = sample(labels[batch], n_classes)
        return z

    def get_prior(self, prior_type):
        if prior_type == "gaussian_mixture":
            return self.gaussian_mixture
        elif prior_type == "swiss_roll":
            return self.swiss_roll
        else:
            raise ValueError(
                "You passed in prior_type={}, supported types are: "
                "gaussian_mixture, swiss_roll".format(prior_type)
            )
