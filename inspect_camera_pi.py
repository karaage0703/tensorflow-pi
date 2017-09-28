#!/usr/bin/env python
#! -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import numpy as np
import cv2
import picamera
from time import sleep

photo_filename = '/tmp/data.jpg'

def shutter():
    photofile = open(photo_filename, 'wb')
    print(photofile)

    with picamera.PiCamera() as camera:
        camera.resolution = (640,480)
        camera.start_preview()
        sleep(1.000)
        camera.capture(photofile)

if __name__ == '__main__':
    # inport Inception V3 model
    model = InceptionV3(weights='imagenet')

    cam = cv2.VideoCapture(0)
    while True:
        shutter()
        img = cv2.imread(photo_filename)
        img = cv2.resize(img, (299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)

        print('Predicted:')
        for p in decode_predictions(preds, top=5)[0]:
            print("Score {}, Label {}".format(p[2], p[1]))
        count = 0
