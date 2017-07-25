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
$ cd
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
Then increase data(mirror, rotate,...) if you need.

If you would like to increase panda photo, execute following commands:
```sh
$ cd ~/tensorflow_pi/data/image_dir/panda
$ ../../increase_picture-all.sh
```


#### Make label data
Execute following commands for making label data:
```sh
$ cd ~/tensorflow_pi/data
$ python make_train_data.py image_dir
```

This commands display number of class.

Modify `NUM_CLASSES` of `cnn.py`

## Train
Execute following commands for train:
```sh
$ cd ~/tensorflow_pi
$ python train.py
```

## Test
Execute following commands for test:
```sh
$ cd ~/tensorflow_pi
$ python predict.py <imagefilename>
```

ex:
```sh
$ python predict.py giantpanda.jpg
```

```
panda
```

## Visualization(Tensor Board)
Execute following commnads after train:
```sh
$ tensorboard --logdir=/tmp/tensorflow_pi/
```

Then access raspberry pi. Check your ip adress by executing `ifconfig` command. And access following address.
```
<ip address>:6006
```

For example
```
192.168.0.10:6006
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

