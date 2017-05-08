import os
import random

import images_to_numpy as iton
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
            return a #self.batch_normalization(a, is_training, collection, conv_name+'_batchnorm')

    def pool(self, x, f_shape, pool_name):
        with tf.variable_scope(pool_name):
            return tf.nn.max_pool3d(x, f_shape, f_shape, "SAME", name=pool_name+'_pool')

    def unpool(self, x, unpool_name):
        with tf.variable_scope(unpool_name):
            return tf.concat([x, x], axis=1)

    def last_conv(self, x, w_shape, b_shape, is_training, collection, conv_name):
        return self.conv(x, w_shape, b_shape, is_training, collection, conv_name, last=True)

    def deconv(self, x, w_shape, b_shape, is_training, collection, deconv_name):
        with tf.variable_scope(deconv_name):
            weights = self.weight_variable(w_shape, collection, deconv_name)
            out_shape = tf.stack([tf.shape(x)[0], w_shape[0], 128, 128, w_shape[4]])
            h = tf.nn.conv3d_transpose(x, weights, out_shape, [1, 1, 1, 1, 1], "SAME",
                                       name=deconv_name+'_deconv')
            a = tf.nn.relu(h + self.bias_variable(b_shape, collection, deconv_name),
                           name=deconv_name+'_relu')
            return self.batch_normalization(a, is_training, collection, deconv_name+'_batchnorm')

    def save(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, self.ckpt_dir)

    def __init__(self):
    
        # toggle training
        self.train = False

        # hyperparams
        self.bn_decay = 0.99
        self.epsilon = 1e-3

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        with tf.device('/cpu:0'):
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            self.train_gen = tf.placeholder(tf.bool)
            self.train_dis = tf.placeholder(tf.bool)
            
            # generator
            g = 'generator'
            with tf.variable_scope(g):
                self.x = tf.placeholder(tf.float32, [None, 1, 128, 128, 3])
                
                self.c1 = self.conv(self.x, [1, 3, 3, 3, 8], [8], self.train_gen, g, 'c1')
                self.c2 = self.conv(self.c1, [1, 3, 3, 8, 16], [16], self.train_gen, g, 'c2')
                self.u1 = self.unpool(self.c2, 'u1')
                #self.d1 = self.deconv(self.c1, [2, 3, 3, 32, 32], [32], self.train_gen, g, 'd1')
                self.c3 = self.conv(self.u1, [2, 3, 3, 16, 32], [32], self.train_gen, g, 'c3')
                self.c4 = self.conv(self.c3, [2, 3, 3, 32, 64], [64], self.train_gen, g, 'c4')
                self.u2 = self.unpool(self.c4, 'u2')
                #self.d2 = self.deconv(self.c2, [4, 3, 3, 64, 64], [64], self.train_gen, g, 'd2')
                self.c5 = self.conv(self.u2, [4, 3, 3, 64, 32], [32], self.train_gen, g, 'c5')
                self.c6 = self.conv(self.c5, [4, 3, 3, 32, 16], [16], self.train_gen, g, 'c6')
                self.u3 = self.unpool(self.c6, 'u3')
                #self.d3 = self.deconv(self.c3, [8, 3, 3, 128, 128], [128], self.train_gen, g, 'd3')
                #self.c4 = self.conv(self.u3, [8, 3, 3, 64, 32], [32], self.train_gen, g, 'c4')
                #self.u4 = self.unpool(self.c4, 'u4')
                #self.d4 = self.deconv(self.c4, [16, 3, 3, 64, 64], [64], self.train_gen, g, 'd4')
                #self.c5 = self.conv(self.u4, [16, 3, 3, 32, 16], [16], self.train_gen, g, 'c5')
                #self.u5 = self.unpool(self.c5, 'u5')
                #self.d5 = self.deconv(self.c5, [32, 3, 3, 32, 32], [32], self.train_gen, g, 'd5')
                self.c7 = self.conv(self.u3, [8, 3, 3, 16, 8], [8], self.train_gen, g, 'c7')
                self.c8 = self.conv(self.c7, [8, 3, 3, 8, 3], [3], self.train_gen, g, 'c8', last=True)
                #tf.summary.histogram('generator_conv4', self.c4)
                self.gen = tf.sigmoid(self.c8)

                self.split = tf.split(self.gen, num_or_size_splits=8, axis=1)
                for i in range(8):
                    tf.summary.image('img_{}'.format(i), tf.reshape(self.split[i], [-1, 128, 128, 3]))

                self.g = tf.concat([self.x, self.gen], axis=1)

            # discriminator
            d = 'discriminator'
            with tf.variable_scope(d):
                self.y = tf.placeholder(tf.float32, [None, 9, 128, 128, 3])
                self.z = tf.concat([self.g, self.y], axis=0)
                self.x_l = tf.placeholder(tf.float32, [None, 1])
                self.y_l = tf.placeholder(tf.float32, [None, 1])
                self.l = tf.concat([self.x_l, self.y_l], axis=0)

                self.h1 = self.conv(self.z, [3, 3, 3, 3, 16], [16], self.train_dis, d, 'h1')
                self.p1 = self.pool(self.h1, [1, 2, 2, 2, 1], 'p1')
                self.h2 = self.conv(self.p1, [3, 3, 3, 16, 32], [32], self.train_dis, d, 'h2')
                self.p2 = self.pool(self.h2, [1, 2, 2, 2, 1], 'p2')
                self.h3 = self.conv(self.p2, [3, 3, 3, 32, 16], [16], self.train_dis, d, 'h3')
                self.p3 = self.pool(self.h3, [1, 2, 2, 2, 1], 'p3')
                self.h4 = self.conv(self.p3, [3, 3, 3, 16, 8], [8], self.train_dis, d, 'h4')
                self.p4 = self.pool(self.h4, [1, 2, 4, 4, 1], 'p4')
                self.h5 = self.conv(self.p4, [3, 3, 3, 8, 1], [1], self.train_dis, d, 'h5', last=True)
                #tf.summary.histogram('discriminator_conv5', self.h5)
                self.p5 = self.pool(self.h5, [1, 3, 4, 4, 1], 'p5')

                self.dis = tf.reshape(self.p5, [-1, 1])
                tf.summary.histogram('discriminator_prediction', self.dis)
                self.dg = tf.slice(self.dis, [0, 0], [tf.shape(self.g)[0], 1])

            with tf.variable_scope('loss'):
                self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis,
                                               labels=self.l))
                #self.dis_loss = -tf.reduce_mean(self.l*tf.log(self.dis+1e-8) +
                #                               (1-self.l)*tf.log(1-self.dis+1e-8))
                self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dg,
                                               labels=1-self.x_l))
                #self.gen_loss = -tf.reduce_mean((1-self.x_l)*tf.log(self.dg+1e-8) + 
                #                                self.x_l*tf.log(1-self.dg+1e-8))
                tf.summary.scalar('discriminator_loss', self.dis_loss)
                tf.summary.scalar('generator_loss', self.gen_loss)

                self.dis_step = tf.train.AdamOptimizer(1e-4).minimize(self.dis_loss,
                                                                      var_list=tf.get_collection(d))
                self.gen_step = tf.train.AdamOptimizer(1e-3).minimize(self.gen_loss,
                                                                      var_list=tf.get_collection(g))
            
            self.saver = tf.train.Saver()
            self.f_dir = 'frameonly'
            self.ckpt_dir = self.f_dir + '/checkpoints'
            self.log_dir = self.f_dir + '/logs'
            if not self.train:
                latest_ckpt = tf.train.latest_checkpoint(self.f_dir)
                print('Loading {}'.format(latest_ckpt))
                self.saver.restore(self.sess, latest_ckpt)
                self.merged = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
            else:
                self.merged = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
                self.sess.run(tf.global_variables_initializer())

model = Model()
data_files = os.listdir('vehicles_auto_fps_5')
train_idx = [i for i in range(500)]
val_idx = [i for i in range(500, len(data_files))]
num_epochs = 100
step = 0
max_batch_size = 60
if model.train:
    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch+1))
        random.shuffle(train_idx)
        for i in range(len(train_idx)):
            step += 1

            if not step % 3 == 0:
                frames = iton.images_to_numpy('vehicles_auto_fps_5/'+data_files[train_idx[i]]+'/')
                # train generator
                x = []
                y = []
                fr = [l for l in range(len(frames)-9)]
                if len(frames)-9 > max_batch_size:
                    random.shuffle(fr)
                    fr = fr[:max_batch_size]
                if len(fr) != 0:
                    for j in fr:
                        x.append(np.array([frames[j]]))
                    x = np.array(x)
                    y = np.zeros((0, 9, 128, 128, 3))
                    x_l = np.zeros((x.shape[0], 1))
                    y_l = np.ones((0, 1))
                    fd = {model.x:x, model.y:y, model.x_l:x_l, model.y_l:y_l,
                          model.train_gen:True, model.train_dis:False}
                    model.sess.run(model.gen_step, feed_dict=fd)

            else:
                if epoch > 20 and step % 2 == 0:
                    x = []
                    y = []
                    fr = [l for l in range(len(frames)-9)]
                    if len(frames)-9 > max_batch_size:
                        random.shuffle(fr)
                        fr = fr[:max_batch_size]
                    if len(fr) != 0:
                        bad_num = len(fr) // 2
                        for j in fr:
                            y_idx = []
                            if j < bad_num // 2:
                                for k in range(9):
                                    y_idx.append(frames[j])
                            else:
                                for k in range(j, j+9):
                                    y_idx.append(frames[k])
                                if j < bad_num:
                                    random.shuffle(y_idx)
                            y.append(np.array(y_idx))
                        x = np.ones((0, 1, 128, 128, 3))
                        y = np.array(y)
                        x_l = np.zeros((bad_num, 1))
                        y_l = 0.9 * np.ones((y.shape[0] - bad_num, 1))
                        fd = {model.x:x, model.y:y, model.x_l:x_l, model.y_l:y_l,
                              model.train_gen:False, model.train_dis:True}
                        model.sess.run(model.dis_step, feed_dict=fd)

                else:
                    x = []
                    y = []
                    fr = [l for l in range(len(frames)-9)]
                    if len(frames)-9 > max_batch_size // 2:
                        random.shuffle(fr)
                        fr = fr[:max_batch_size // 2]
                    if len(fr) != 0:
                        for j in fr:
                            x.append(np.array([frames[j]]))
                            y_idx = []
                            for k in range(j, j+9):
                                y_idx.append(frames[k])
                            y.append(np.array(y_idx))
                        x = np.array(x)
                        y = np.array(y)
                        x_l = np.zeros((x.shape[0], 1))
                        y_l = 0.9 * np.ones((y.shape[0], 1))
                        fd = {model.x:x, model.y:y, model.x_l:x_l, model.y_l:y_l,
                              model.train_gen:False, model.train_dis:True}
                        model.sess.run(model.dis_step, feed_dict=fd)
            
            if step % 10 == 0:
                x = []
                y = []
                fr = [l for l in range(len(frames)-9)]
                if len(frames)-9 > max_batch_size // 2:
                    random.shuffle(fr)
                    fr = fr[:max_batch_size // 2]
                if len(fr) != 0:
                    for j in fr:
                        x.append(np.array([frames[j]]))
                        y_idx = []
                        for k in range(j, j+9):
                            y_idx.append(frames[k])
                        y.append(np.array(y_idx))
                    x = np.array(x)
                    y = np.array(y)
                    x_l = np.zeros((x.shape[0], 1))
                    y_l = np.ones((y.shape[0], 1))
                    fd = {model.x:x, model.y:y, model.x_l:x_l, model.y_l:y_l,
                          model.train_gen:False, model.train_dis:True}
                    summary = model.sess.run(model.merged, feed_dict=fd)
                    model.writer.add_summary(summary, step)
                model.save()
            
else:
    lst = [train_idx[0], val_idx[0]]
    for idx in lst:
        frames = iton.images_to_numpy('vehicles_auto_fps_5/'+data_files[idx]+'/')
        start = random.randint(0, len(frames)-9)
        x = np.array([[frames[start]]])
        fd = {model.x:x, model.train_gen:False}
        pred_video = model.sess.run(model.g, feed_dict=fd)
        np.save('generated_video/'+data_files[idx], pred_video)
