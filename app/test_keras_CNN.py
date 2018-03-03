import io
import os
import pickle
import sys
import numpy as np
import time

#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

import keras
from keras import backend as K
from keras.datasets import mnist

# print devices
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#


#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


# <<< set path and import
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(current_dir)
data_dir = os.path.join(top_dir, 'data')
sys.path.append(top_dir)
from ANNs.CNN.cnn_keras import CNNKeras
# <<<

# ----------------------------------------------------------------------------------------------------------------------
# CNN config & build CNN
# ----------------------------------------------------------------------------------------------------------------------
# config
img_rows = 28
img_cols = 28
input_shape = (img_rows,img_cols,1)
batch_size = 100
num_classes = 10
epochs = 5


# build cnn
cnn_keras = CNNKeras(input_shape=input_shape, num_classes=num_classes)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# set path & read dataset
# ----------------------------------------------------------------------------------------------------------------------
# read dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# reshape
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

print('x_train shape:', x_train.shape) # (60000, 28, 28)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices [5 0 4 ... 5 6 8] -> [[[1. 0. 0. ... 0. 0. 0.]],...] one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------
cnn_keras.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# testing
# ----------------------------------------------------------------------------------------------------------------------
score = cnn_keras.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# ----------------------------------------------------------------------------------------------------------------------
