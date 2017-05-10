from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import csv
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.utils import np_utils
from keras import backend as K

#Training Parameters
batch_size = 200
epochs = 12

#11 Activities
num_classes = 11

#Model HyperParameters
filter_size= 20
max_pooling_size = 3
dropout_probability = (0.7)
learning_rate = 0.05
momentum = 0.5
decay = 0.25

model = Sequential()

#X
model.add(Dense(64))
model.add(Conv2D(12 + filter_size + 1, filter_size))
model.add(Activation('tanh'))
model.add(MaxPooling2D((4, max_pooling_size)))

#Y
model.add(Dense(64))
#N - m + 1
model.add(Conv2D(12 + filter_size + 1, filter_size))
model.add(Activation('tanh'))
model.add(MaxPooling2D((4, max_pooling_size)))

#Z
model.add(Dense(64))
model.add(Conv2D(12 + filter_size + 1, filter_size))
model.add(Activation('tanh'))
model.add(MaxPooling2D((4, max_pooling_size)))

#Hidden Layers
model.add(Dense(1024))
model.add(Dense(30))
Dropout(dropout_probability)

#SoftMax
model.add(Activation('softmax'))

#Forward Propogation

#Backwards Propogation

#Regularization
keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)



