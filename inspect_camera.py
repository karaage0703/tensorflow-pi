#!/usr/bin/env python
#! -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import numpy as np
import cv2

if __name__ == '__main__':
    # inport Inception V3 model
    model = InceptionV3(weights='imagenet')

    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        cv2.imshow('tensorflow-pi inspector', capture)
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count == 30:
            img = capture.copy()
            img = cv2.resize(img, (299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)

            print('Predicted:')
            for p in decode_predictions(preds, top=5)[0]:
                print("Score {}, Label {}".format(p[2], p[1]))
            count = 0

    cam.release()
    cv2.destroyAllWindows()
