#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

NUM_CLASSES = 4
# IMAGE_SIZE can be multipled by 4 ex: 28,32,36...
IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def inference(images_placeholder, keep_prob):
    """ inference function

    args:
      images_placeholder: image placeholder
      keep_prob: dropout place_holder

    return:
      y_conv:
    """
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1') as scope:
        FC_LAYER_SIZE = IMAGE_SIZE / 2 / 2
        W_fc1 = weight_variable([FC_LAYER_SIZE*FC_LAYER_SIZE*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, FC_LAYER_SIZE*FC_LAYER_SIZE*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # set dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def loss(logits, labels):
    """ loss function

    args:
      logits: logits tensor, float - [batch_size, NUM_CLASSES]
      labels: labels tensor, int32 - [batch_size, NUM_CLASSES]

    return:
      cross_entropy: cross entropy tensor, float

    """

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """ training function

    args:
      loss: loss tensor, loss()の結果
      learning_rate: learning rate

    return:
      train_step: 訓練のOp

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ accuracy function

    args:
      logits: results of inference()
      labels: labels tensor, int32 - [batch_size, NUM_CLASSES]

    return:
      accuracy: accyracy(float)

    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
