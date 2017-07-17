# tensorflow-pi
Useful tools for using tensorflow easily on Mac/Linux(including Raspberry Pi)

# Dependency

- tensorflow 1.2.1
- numpy
- OpenCV2
- python 2.7

# Setup
## Mac

## Linux


## Raspberry Pi

# Usage

## Clone this repository
```sh
$ git clone https://github.com/karaage0703/tensorflow_pi
$ cd tensorflow_pi
```

## Prepare your own data and make label data
### Save your Image files which classified
Save your Image files like belows:

```
tensorflow-pi
 |- data/
     image_dir/
       |--- panda/giantpanda.jpg, panda.jpg, ...
       |--- flower/dandelion.jpg, rose.jpg, ...
       |--- ...
       |--- ...
```

#### Increase data
Then increase data(mirror, rotate,...) if you need:


#### Make label data
Execute following commands for making label data:
```sh
$ cd data
$ python make_train_data.py image_dir
```

### Change CLASS NUMBER
```sh
$ vim cnn.py
```

change class number
## Train

```sh
$ python train.py
```

## Test
```sh
$ python test.py <imagefilename>
```

ex:

```sh
$ python cnn_test.py flower.py
```

```
flower
```


# License
This software is released under the MIT License, see LICENSE.
However some programs lisenses are not MIT. Please confirm `Authors` in this README.md.

# Authors
Following programs original file links are listed. These programs is released under original file lisense.
Please confirm lisense of original file if you need.

- [uei](https://github.com/uei/deel) (make_train_data.py)
- [google](https://github.com/tensorflow/tensorflow) (cnn*.py)
- [bohemian916](https://github.com/bohemian916/deeplearning_tool) (increase_picture.py)

# References
- https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py
- http://qiita.com/shu223/items/ef160cbe1e9d9f57c248
- http://kivantium.hateblo.jp/entry/2015/11/18/233834

