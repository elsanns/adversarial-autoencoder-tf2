# coding=utf-8
""""Functions used for model training. Training at the batch level performed in a static graph: 
function: train_step decorated with `@tf.function` decorator."""


import numpy as np
import tensorflow as tf
from model.aae import Gan, Encoder, Discriminator
from utils.data_loader import DataLoader
from utils.prior_utils import PriorFactory
from utils.plot_utils import PlotFactory
import os
import datetime


@tf.function
def train_step(x_batch, y_labels, gan, optimizers_dict, label_sample, real_distribution, n_classes):
    """"Training of one batch execute in graph mode."""
    
    # Gan
    with tf.GradientTape() as gan_tape:
        x_reconstruction = gan.decoder(gan.encoder(x_batch))
        gan_loss = Gan.get_loss(x_batch, x_reconstruction)

    trainable_variables = gan.encoder.trainable_variables + gan.decoder.trainable_variables
    gan_gradients = gan_tape.gradient(gan_loss, trainable_variables)
    optimizers_dict['gan'].apply_gradients(zip(gan_gradients, trainable_variables))

    # Discriminator
    with tf.GradientTape() as discriminator_tape:
        label_sample_one_hot = tf.one_hot(label_sample, n_classes)
        real_distribution_label = tf.concat([real_distribution, label_sample_one_hot], axis=1)

        fake_distribution = gan.encoder(x_batch)
        fake_distribution_label = tf.concat([fake_distribution, y_labels], axis=1)

        _, real_logits = gan.discriminator(real_distribution_label)
        _, fake_logits = gan.discriminator(fake_distribution_label)
        discriminator_loss = Discriminator.get_loss(real_logits, fake_logits)

    discriminator_gradients = discriminator_tape.gradient(discriminator_loss, gan.discriminator.trainable_variables)
    optimizers_dict['discriminator'].apply_gradients(
        zip(discriminator_gradients, gan.discriminator.trainable_variables))

    # Encoder
    with tf.GradientTape() as encoder_tape:
        encoder_output = gan.encoder(x_batch)
        encoder_output_label = tf.concat([encoder_output, y_labels], axis=1)
        _, disc_fake_logits = gan.discriminator(encoder_output_label)
        encoder_loss = Encoder.get_loss(disc_fake_logits)

    encoder_gradients = encoder_tape.gradient(encoder_loss, gan.encoder.trainable_variables)
    optimizers_dict['encoder'].apply_gradients(zip(encoder_gradients, gan.encoder.trainable_variables))

    return gan_loss, discriminator_loss, encoder_loss


def train_all_steps(gan, optimizers_dict, train_ds, n_epochs, prior_type, n_classes, data_loader,
                    plot_factory, log_dir):
    """"Training of all batches `n_epochs` of times"""
    
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    gan_loss_vec = tf.metrics.Mean()
    discriminator_loss_vec = tf.metrics.Mean()
    encoder_loss_vec = tf.metrics.Mean()
    total_loss_vec = tf.metrics.Mean()

    for epoch in range(n_epochs):
        for batch_no, (x_batch, y_labels) in enumerate(train_ds):
            label_sample = np.random.randint(0, n_classes, size=[x_batch.shape[0]])
            real_distribution = plot_factory.prior_factory.get_prior(prior_type)(x_batch.shape[0], label_sample, n_classes)

            gan_loss, discriminator_loss, encoder_loss = train_step(x_batch, y_labels, gan, optimizers_dict,
                                                                    label_sample, real_distribution, n_classes)

            gan_loss_vec(gan_loss)
            discriminator_loss_vec(discriminator_loss)
            encoder_loss_vec(encoder_loss)
            total_loss = gan_loss + discriminator_loss + encoder_loss
            total_loss_vec(total_loss)

            with summary_writer.as_default():
                tf.summary.scalar('gan_loss', gan_loss_vec.result(), step=optimizers_dict['gan'].iterations)
                tf.summary.scalar('encoder_loss', encoder_loss_vec.result(), step=optimizers_dict['encoder'].iterations)
                tf.summary.scalar('discriminator_loss', discriminator_loss_vec.result(),
                                  step=optimizers_dict['discriminator'].iterations)

        print("Epoch: {} total_loss: {} gan_loss: {}, discriminator_loss: {} encoder_loss: {}".
              format(epoch,
                     total_loss_vec.result(),
                     gan_loss_vec.result(),
                     discriminator_loss_vec.result(),
                     encoder_loss_vec.result()))

        visualize_results(gan, data_loader, plot_factory, n_classes, prior_type, epoch)


def visualize_results(gan, data_loader, plot_factory, n_classes, prior_type, epoch=None):
    """Creates plots visualizing training results: reconstruction, latent code distribution 
    and generator output for point sampled from the latent space. Plotting is handled by the plot_factory object."""
    
    n_tot_imgs = n_classes * n_classes
    n_tot_imgs_sampled = plot_factory.x_sampling_reconstr * plot_factory.y_sampling_reconstr
    dist_sample_count = 10000

    # Distribution demo
    plot_factory.plot_distribution_demo(prior_type, dist_sample_count)

    # Test data for reconstruction
    x_test, _ = data_loader.get_test_sample(n_tot_imgs, n_tot_imgs)
    x_test_img = x_test.reshape(n_tot_imgs, data_loader.img_size_x, data_loader.img_size_y)
    plot_factory.save_images(x_test_img, name="x_input.png")

    # Test data for distribution plot
    x_dist, id_dist = data_loader.get_test_sample(dist_sample_count, dist_sample_count)

    # Test data for sampling from distribution
    z_samples = plot_factory.plot_sampling_reconstr()

    # Plot reconstruction
    y_reconstruction = gan.decoder(gan.encoder(x_test, training=False), training=False)
    y_reconstruction_img = tf.reshape(y_reconstruction,
                                      (n_tot_imgs, data_loader.img_size_x, data_loader.img_size_y)).numpy()
    plot_factory.save_images(y_reconstruction_img, name="/x_reconstruction_epoch_{}.png".format(epoch))

    if prior_type == 'gaussian_mixture':
        # Plot decoder output for z samples
        y_for_z_samples = gan.decoder(z_samples, training=False)
        y_for_z_samples_img = tf.reshape(y_for_z_samples,
                                         (n_tot_imgs_sampled, data_loader.img_size_x, data_loader.img_size_y)).numpy()
        plot_factory.save_images_sampled(y_for_z_samples_img, name="z_sampled_reconstr_{}.png".format(epoch))

    # z distribution by label
    z_dist = gan.encoder(x_dist, training=False)
    plot_factory.plot_distribution(z_dist, id_dist, name="/z_distribution_{}.png".format(epoch))


def train_model(args):
    result_dir = args.results_dir
    log_dir = args.log_dir
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    prior_type = args.prior_type
    n_classes = args.n_classes
    gm_x_stddev = args.gm_x_stddev
    gm_y_stddev = args.gm_y_stddev

    # Data
    data_loader = DataLoader(batch_size)
    train_ds, test_ds = data_loader.make_dataset()

    # Prior and Plot objects
    prior_factory = PriorFactory(n_classes, gm_x_stddev=gm_x_stddev, gm_y_stddev=gm_y_stddev)
    plot_factory = PlotFactory(prior_factory, result_dir, prior_type, n_classes)

    # Model
    gan = Gan(image_dim=data_loader.img_size_x*data_loader.img_size_y)

    # Optimizers
    optimizers_dict = {'encoder': tf.optimizers.Adam(learning_rate=learning_rate),
                       'discriminator': tf.optimizers.Adam(learning_rate=learning_rate/5),
                       'gan': tf.optimizers.Adam(learning_rate=learning_rate)}

    # Training
    train_all_steps(gan, optimizers_dict, train_ds, n_epochs, prior_type, n_classes, data_loader,
                    plot_factory, log_dir)
