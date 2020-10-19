from utils import parse_args, create_experiment_dirs, calculate_flops
from reduced_model import TinyNet
from train import Train
from data_loader import DataLoader
from summarizer import Summarizer
import tensorflow as tf
import sys


def main(args):
    t = args[0]
    rho = 2
    phi = 10
    if len(args) == 3:
      rho = args[1]
      phi = args[2]

    # Parse the JSON arguments
    config_args = parse_args()
    config_args['experiment_dir'] = config_args['experiment_dir'] + '_' + t

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = TinyNet(config_args, rho, phi)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()

    if config_args.to_test:
        print("Final test!")
        trainer.test('val')
        print("Testing Finished\n\n")


main(sys.argv[1:])
