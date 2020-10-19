from easydict import EasyDict as edict
import json
import argparse
import os
import pickle
import tensorflow as tf
from pprint import pprint
import sys


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """

    config_args_dict = {
          "experiment_dir": "TinyNet_v5_train_experiment",
          "num_epochs": 100,
          "num_classes": 2,
          "batch_size": 32,
          "width_multiplier": 1.0,
          "shuffle": True,
          "l2_strength": 4e-5,
          "bias": 0.0,
          "learning_rate": 1e-3,
          "batchnorm_enabled": True,
          "dropout_keep_prob": 0.999,
          "pretrained_path": "pretrained_weights/mobilenet_v1.pkl",
          "max_to_keep": 4,
          "save_model_every": 5,
          "test_every": 5,
          "to_train": True,
          "to_test": False
        }

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
