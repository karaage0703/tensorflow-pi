#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import cnn as nn
from os import path
import picamera
from time import sleep

face_filename = '/tmp/face.jpg'
cascades_dir ='/usr/share/opencv/haarcascades'

def shutter():
    photofile = open(face_filename, 'wb')
    print(photofile)

    with picamera.PiCamera() as camera:
        camera.resolution = (640,480)
        camera.start_preview()
        sleep(1.000)
        camera.capture(photofile)

def face_detect():
    image = cv2.imread(face_filename)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_f = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_frontalface_alt2.xml'))
    cascade_e = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_eye.xml'))

    facerect = cascade_f.detectMultiScale(image_gray, scaleFactor=1.08, minNeighbors=1, minSize=(200, 200))

    # print("face rectangle")
    # print(facerect)

    image_face = []
    if len(facerect) > 0:
        # filename numbering
        numb = 0
        tmp_size = 0
        for rect in facerect:
            x, y, w, h = rect
            # eyes in face?
            roi = image_gray[y: y + h, x: x + w]
            eyes = cascade_e.detectMultiScale(roi, scaleFactor=1.05, minSize=(20,20))
            if len(eyes) > 1:
                if h > tmp_size:
                    tmp_size = h
                    image_face = image[y:y+h, x:x+h]

    return image_face

if __name__ == '__main__':
    labels = []
    backup_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/model"
    f = open(backup_dir + '/labels.txt', 'r')

    print(f)
    for line in f:
        labels.append(line.rstrip())

    # load network
    images_placeholder = tf.placeholder("float", shape=(None, nn.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, nn.NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = nn.inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, backup_dir + "/model.ckpt")

    try:
        while True:
            shutter()
            image_face = face_detect()
            if image_face != [] :
                print("Face detected!")
                # cv2.imwrite("/tmp/face_detected.jpg", image_face)

                img = cv2.resize(image_face, (nn.IMAGE_SIZE, nn.IMAGE_SIZE))
                test_image = img.flatten().astype(np.float32)/255.0
                test_image = np.asarray(test_image)

                pred = np.argmax(logits.eval(feed_dict={
                    images_placeholder: [test_image],
                    keep_prob: 1.0 })[0])
                print(labels[pred])

    finally:
        print("program terminated")
        sess.close()
