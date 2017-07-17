#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2

import cnn as nn

if __name__ == '__main__':
    test_image = []
    for i in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[i])
        img = cv2.resize(img, (nn.IMAGE_SIZE, nn.IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, nn.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, nn.NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = nn.inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model.ckpt")

    for i in range(len(test_image)):
        pred = np.argmax(logits.eval(feed_dict={ 
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0])
        print pred
