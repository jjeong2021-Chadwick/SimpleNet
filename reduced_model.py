import tensorflow as tf
from layers import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout
import os
from utils import load_obj, save_obj
import numpy as np


class TinyNet:
    """
    MobileNet Class
    """

    def __init__(self,
                 args, rho=2, phi=10):

        # init parameters and input
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None
        self.args = args
        self.mean_img = None
        self.nodes = dict()
        self.rho = rho
        self.phi = phi

        self.pretrained_path = os.path.realpath(self.args.pretrained_path)

        self.__build()

    def __init_input(self):
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [self.args.batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels],
                                    name='input_tensor')
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.placeholder(tf.int32, [self.args.batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)

    def __init_mean(self):
        # Preparing the mean image.
        img_mean = np.ones((1, 224, 224, 3))
        img_mean[:, :, :, 0] *= 103.939
        img_mean[:, :, :, 1] *= 116.779
        img_mean[:, :, :, 2] *= 123.68
        self.mean_img = tf.constant(img_mean, dtype=tf.float32)

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_mean()
        self.__init_input()
        self.__init_network()
        self.__init_output()

    def __init_network(self):
        with tf.variable_scope('mobilenet_encoder'):
            # Preprocessing as done in the paper
            with tf.name_scope('pre_processing'):
                preprocessed_input = (self.X - self.mean_img) / 255.0

            # Model is here!
            conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * self.args.width_multiplier)),
                             kernel_size=(3, 3),
                             padding='SAME', stride=(2, 2), activation=tf.nn.relu6,
                             batchnorm_enabled=self.args.batchnorm_enabled,
                             is_training=self.is_training, l2_strength=self.args.l2_strength, bias=self.args.bias)
            self.__add_to_nodes([conv1_1])
            ############################################################################################
            out = conv1_1
            counter = 2
            for p in range(6, self.phi + 1):
                layer_name = 'conv_ds_' + str(counter)
                filters = 2 ** p
                stride = (2, 2)
                if p == 6:
                    stride = (1, 1)
                dw, pw = depthwise_separable_conv2d(layer_name, out,
                                                    width_multiplier=self.args.width_multiplier,
                                                    num_filters=filters, kernel_size=(3, 3), padding='SAME',
                                                    stride=stride,
                                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                                    activation=tf.nn.relu6,
                                                    is_training=self.is_training,
                                                    l2_strength=self.args.l2_strength,
                                                    biases=(self.args.bias, self.args.bias))
                counter += 1
                out = pw
                self.__add_to_nodes([dw, pw])

                if self.rho > 1:
                    layer_name = 'conv_ds_' + str(counter)
                    dw, pw = depthwise_separable_conv2d(layer_name, out,
                                                        width_multiplier=self.args.width_multiplier,
                                                        num_filters=filters, kernel_size=(3, 3), padding='SAME',
                                                        stride=(1, 1),
                                                        batchnorm_enabled=self.args.batchnorm_enabled,
                                                        activation=tf.nn.relu6,
                                                        is_training=self.is_training,
                                                        l2_strength=self.args.l2_strength,
                                                        biases=(self.args.bias, self.args.bias))
                    counter += 1
                    out = pw
                    self.__add_to_nodes([dw, pw])
            ############################################################################################
            avg_pool = avg_pool_2d(out, size=(7, 7), stride=(1, 1))
            dropped = dropout(avg_pool, self.args.dropout_keep_prob, self.is_training)
            self.logits = flatten(conv2d('fc', dropped, kernel_size=(1, 1), num_filters=self.args.num_classes,
                                         l2_strength=self.args.l2_strength,
                                         bias=self.args.bias))
            self.__add_to_nodes([avg_pool, dropped, self.logits])


    def __init_output(self):
        with tf.variable_scope('output'):
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='loss'))
            self.loss = self.regularization_loss + self.cross_entropy_loss

            # Important for Batch Normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)
            self.y_out_argmax = tf.argmax(tf.nn.softmax(self.logits), axis=-1, output_type=tf.int32)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))

        # Summaries needed for TensorBoard
        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()

    def __restore(self, file_name, sess):
        try:
            print("Loading ImageNet pretrained weights...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet_encoder')
            dict = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            sess.run(run_list)
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        self.__restore(self.pretrained_path, sess)

    def __add_to_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
