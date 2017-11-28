#!/usr/bin/env python
# encoding: utf-8

# referred from: https://github.com/llSourcell/Pokemon_GAN/blob/master/pokeGAN.py

import os
import tensorflow as tf
import numpy as np
from utils import save_images
from datetime import datetime

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
version = 'newPokemon'
newPoke_path = './' + version
if not os.path.exists(newPoke_path):
    os.mkdir(newPoke_path)


def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)


def process_data():
    # image files => content queue
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, 'resized_black')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir, each))
    # read content and convert into string
    all_images = tf.convert_to_tensor(images, dtype=tf.string)
    images_queue = tf.train.slice_input_producer([all_images])

    # load image content and add noises
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels=CHANNEL)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT, WIDTH, CHANNEL])

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE,
                                          num_threads=4, capacity=200 + 3 * BATCH_SIZE,
                                          min_after_dequeue=200)
    num_images = len(images)
    return images_batch, num_images


def generator(input, random_dim, is_training, reuse=False):
    """
    batch_size X random_dim ==> batch_size X (128 X 128 X 3) with deconv layers
    """
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32   # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB, so 3
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        # 1. random dim input ==> s4 * s4 * c4
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        # 2. reshape to 4 X 4 images with 512 channels
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')

        # 3. 4 X 4 X 512 => 8 X 8 X 256.  conv2d_transpose means DECONV, so with "same" padding and strides=2, end up 8 X 8
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')

        # 4. 8 X 8 X 256 => 16 X 16 X 128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')

        # 5. 16 X 16 X 128 => 32 X 32 X 64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')

        # 6. 32 X 32 X 64 => 64 X 64 X 32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        # 7. 64 X 64 X 32 => 128 X 128 X 3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv6')
        act6 = tf.nn.relu(conv6, name='act6')
        return act6


def discriminator(input, is_training, reuse=False):
    """
    batch_size X (128 X 128 X 3) => batch_size X 1 with deconv layers
    """
    c2, c4, c8, c16 = 64, 128, 256, 512   # channel num
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        # 1. 128 X 128 X 3 images => 64 X 64 X 64
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn1')
        act1 = lrelu(bn1, n='act1')

        # 2. => 32 X 32 X 128
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')

        # 3. => 16 X 16 X 256
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')

        # 4. => 8 X 8 X 512
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training, epsilon=1e-5,
                                           decay=0.9, updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # 5. reshape to 8 * 8 * 512
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        # 6. fc layer, 8 * 8 * 512 => 1
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        # dcgan
        # acted_out = tf.nn.sigmoid(logits)
        return logits  # , acted_out


def train():
    random_dim = 100
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_training = tf.placeholder(tf.bool, name='is_training')

    fake_image = generator(random_input, random_dim, is_training)
    real_result = discriminator(real_image, is_training)
    fake_result = discriminator(fake_image, is_training, reuse=True)

    # fake images score low and real ones score high
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    # fake images score high
    g_loss = -tf.reduce_mean(fake_result)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights, which is actually WGAN
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    image_batch, samples_num = process_data()
    batch_num = int(samples_num / BATCH_SIZE)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    save_path = saver.save(sess, './pokegan.ckpt')
    if not os.path.exists('./model/'):
        os.mkdir('./model/')
    if not os.path.exists('./model/' + version):
        os.mkdir('./model/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num: {}'.format(samples_num))
    print('batch size: {}, batch num per epoch: {}, epoch num:{}'.format(BATCH_SIZE, batch_num, EPOCH))
    for i in range(EPOCH):
        print("--------------------------------------")
        print("{} Epoch {} ...".format(str(datetime.now()), i))
        for j in range(batch_num):
            print("batch {} ...".format(j))
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print("d iter {}".format(k))
                # 仍然使用 feed dict 方式，故此需要先把 image batch run 出来
                train_image = sess.run(image_batch)
                # 即使是首次训练时，也要进行 clip
                sess.run(d_clip)
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_training: True})

            for k in range(g_iters):
                print("g iter {}".format(k))
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_training: True})

        if (i + 1) % 250 == 0:
            saver.save(sess, './model/' + version + '/' + str(i + 1))

            sample_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_training: False})
            save_images(imgtest, [8, 8], newPoke_path + '/epoch' + str(i + 1) + '.jpg')
            print('train epoch: [{}], d_loss:{}, g_loss:{}'.format(i, dLoss, gLoss))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    train()
