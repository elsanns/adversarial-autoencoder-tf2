import argparse
from model import trainer


def parse_args():
    desc = "Adversarial autoencoder TensorFlow 2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--prior_type', type=str, default='gaussian_mixture',
                        choices=['gaussian_mixture', 'swiss_roll'], help='Type of target prior distribution', required=True)
    parser.add_argument('--results_dir', type=str, default='results', help='Training visualization directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory (Tensorboard)')
    parser.add_argument('--gm_x_stddev', type=float, default=0.5,
                        help='Gaussian mixture prior: standard dev for the x coord')
    parser.add_argument('--gm_y_stddev', type=float, default=0.1,
                        help='Gaussian mixture prior shape: standard dev for the y coord')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (for further use)')

    return parser.parse_args()


def main(args):
    trainer.train_model(args)


if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()

    main(args)


