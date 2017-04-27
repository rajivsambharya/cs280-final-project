import numpy as np
import tensorflow as tf

class Model:

    def weight_variable(self, shape, collection, layer_name):
        var = tf.get_variable(layer_name+'_weights', shape, 
                              initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(collection, var)
        return var

    def bias_variable(self, shape, collection, layer_name):
        var = tf.get_variable(layer_name+'_bias', shape, initializer=tf.zeros_initializer())
        tf.add_to_collection(collection, var)
        return var

    def batch_normalization(self, x, is_training, bn_name, collection, conv=True):
        with tf.variable_scope(bn_name):
            scale = tf.get_variable(bn_name+'_scale', x.get_shape()[1:],
                                    initializer=tf.ones_initializer())
            offset = tf.get_variable(bn_name+'_offset', x.get_shape()[1:],
                                     initializer=tf.zeros_initializer())
            tf.add_to_collection(collection, scale)
            tf.add_to_collection(collection, offset)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2]) if conv else tf.nn.moments(x, [0])
            ema = tf.train.ExponentialMovingAverage(decay=self.bn_decay)
            def update_distribution():
                ema_apply = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(is_training, update_distribution,
                                lambda: (ema.average(batch_mean),ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, offset, scale, self.epsilon,
                                             name=bn_name+'_bn')

    def conv(self, x, w_shape, b_shape, is_training, collection, conv_name, last=False):
        with tf.variable_scope(conv_name):
            weights = self.weight_variable(w_shape, collection, conv_name)
            h = tf.nn.conv3d(x, weights, [1, 1, 1, 1, 1], "SAME", name=conv_name+'_conv')
            if last:
                return h + self.bias_variable(b_shape, collection, conv_name)
            a = tf.nn.relu(h + self.bias_variable(b_shape, collection, conv_name),
                           name=conv_name+'_relu')
            return self.batch_normalization(a, is_training, collection, conv_name+'_batchnorm')

    def last_conv(self, x, w_shape, b_shape, is_training, collection, conv_name):
        return self.conv(x, w_shape, b_shape, is_training, collection, conv_name, last=True)

    def deconv(self, x, w_shape, b_shape, is_training, collection, deconv_name):
        with tf.variable_scope(deconv_name):
            weights = self.weight_variable(w_shape, collection, deconv_name)
            out_shape = [-1, w_shape[0], 128, 128, w_shape[4]]
            h = tf.nn.conv3d_transpose(x, weights, out_shape, [1, 1, 1, 1, 1], "SAME",
                                       name=deconv_name+'_deconv')
            a = tf.nn.relu(h + self.bias_variable(b_shape, collection, deconv_name),
                           name=deconv_name+'_relu')
            return self.batch_normalization(a, is_training, collection, deconv_name+'_batchnorm')

    def __init__(self):
        # hyperparams
        self.learn_rate = 1e-4
        self.bn_decay = 0.99
        self.epsilon = 1e-3

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.train_gen= tf.placeholder(tf.bool)
        self.train_dis= tf.placeholder(tf.bool)
        
        # generator
        with tf.variable_scope('generator'):
            self.generator_weights = {}
            g = 'generator'
            self.x = tf.placeholder(tf.float32, [None, 1, 128, 128, 3])
            
            self.c1 = self.conv(self.x, [1, 3, 3, 3, 32], [32], self.train_gen, g, 'c1')
            self.d1 = self.deconv(self.c1, [2, 3, 3, 32, 32], [32], self.train_gen, g, 'd1')
            self.c2 = self.conv(self.d1, [2, 3, 3, 32, 64], [64], self.train_gen, g, 'c2')
            self.d2 = self.deconv(self.c2, [4, 3, 3, 64, 64], [64], self.train_gen, g, 'd2')
            self.c3 = self.conv(self.d2, [4, 3, 3, 64, 128], [128], self.train_gen, g, 'c3')
            self.d3 = self.deconv(self.c3, [8, 3, 3, 128, 128], [128], self.train_gen, g, 'd3')
            self.c4 = self.conv(self.d3, [8, 3, 3, 128, 64], [64], self.train_gen, g, 'c4')
            self.d4 = self.deconv(self.c4, [16, 3, 3, 64, 64], [64], self.train_gen, g, 'd4')
            self.c5 = self.conv(self.d4, [16, 3, 3, 64, 32], [32], self.train_gen, g, 'c5')
            self.d5 = self.deconv(self.c5, [32, 3, 3, 32, 32], [32], self.train_gen, g, 'd5')
            self.c6 = self.last_conv(self.d5, [32, 3, 3, 32, 3], [3], self.train_gen, g, 'c6')


model = Model()
"""
if __name__ == "__main__":
    main()
"""
