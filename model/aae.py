# coding=utf-8
"""Classes representing modules of the Adversarial Autoencoder: Encoder, Decoder, Discriminator."""


import tensorflow as tf
from tensorflow.keras import layers, Model


class Encoder(Model):
    """Encoder of the Adversarial Autoencoder model"""
        
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.drop_out_rate = 0.1
        self.n_hidden = 1000
        self.dim_z = 2
        kernel_initializer = tf.initializers.RandomNormal()

        self.dense0 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr0 = layers.ReLU()
        self.drop0 = layers.Dropout(rate=self.drop_out_rate)

        self.dense1 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr1 = layers.ReLU()
        self.drop1 = layers.Dropout(rate=self.drop_out_rate)

        self.z = layers.Dense(self.dim_z, kernel_initializer=kernel_initializer)

    def __call__(self, inputs, training=True):
        x = self.dense0(inputs)
        x = self.lr0(x)
        x = self.drop0(x)
        x = self.dense1(x)
        x = self.lr1(x)
        x = self.drop1(x)
        z = self.z(x)

        return z

    @staticmethod
    def get_loss(disc_fake_logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_fake_logits),
            logits=disc_fake_logits))


class Decoder(Model):
    """Decoder of the Adversarial Autoencoder model"""
        
    def __init__(self, image_dim):
        super(Decoder, self).__init__()
        self.drop_out_rate = 0.1
        self.n_hidden = 1000
        self.image_dim = image_dim
        kernel_initializer = tf.initializers.RandomNormal()

        self.dense0 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr0 = layers.ReLU()
        self.drop0 = layers.Dropout(rate=self.drop_out_rate)

        self.dense1 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr1 = layers.ReLU()
        self.drop1 = layers.Dropout(rate=self.drop_out_rate)

        self.reconstruction = layers.Dense(self.image_dim, activation='sigmoid', kernel_initializer=kernel_initializer)

    def __call__(self, inputs, training=True):
        x = self.dense0(inputs)
        x = self.lr0(x)
        x = self.drop0(x)
        x = self.dense1(x)
        x = self.lr1(x)
        x = self.drop1(x)
        reconstruction = self.reconstruction(x)

        return reconstruction


class Discriminator(Model):
    """Discriminator of the Adversarial Autoencoder model"""
        
    def __init__(self):
        super(Discriminator, self).__init__()
        self.drop_out_rate = 0.1
        self.n_hidden = 1000
        kernel_initializer = tf.initializers.RandomNormal()

        self.dense0 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr0 = layers.ReLU()
        self.drop0 = layers.Dropout(rate=self.drop_out_rate)

        self.dense1 = layers.Dense(self.n_hidden, kernel_initializer=kernel_initializer)
        self.lr1 = layers.ReLU()
        self.drop1 = layers.Dropout(rate=self.drop_out_rate)

        self.prediction_logits = layers.Dense(1, kernel_initializer=kernel_initializer)
        self.prediction = tf.math.sigmoid

    def __call__(self, inputs, training=True):
        x = self.dense0(inputs)
        x = self.lr0(x)
        x = self.drop0(x)
        x = self.dense1(x)
        x = self.lr1(x)
        x = self.drop1(x)
        prediction_logits = self.prediction_logits(x)
        prediction = self.prediction(prediction_logits)

        return prediction, prediction_logits

    @staticmethod
    def get_loss(real_logits, fake_logits):
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_logits), logits=real_logits)
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_logits), logits=fake_logits)

        loss = tf.reduce_mean(loss_real + loss_fake)
        return loss


class Gan(Model):
    """Model of the Adversarial Autoencoder.
    
        Args:
            encoder (Encoder):
                Encoder of the Adversarial Autoencoder (Gan) model.
            decoder (Decoder):
                Decoder of the Adversarial Autoencoder (Gan) model.
            discriminator (Discriminator):
                Discriminator of the Adversarial Autoencoder (Gan) model.
    """

    def __init__(self, image_dim):
        super(Gan, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(image_dim=image_dim)
        self.discriminator = Discriminator()

    @staticmethod
    def get_loss(x_input, x_reconstruction):
        gan_loss = tf.reduce_mean(tf.math.squared_difference(x_input, x_reconstruction))
        return gan_loss
