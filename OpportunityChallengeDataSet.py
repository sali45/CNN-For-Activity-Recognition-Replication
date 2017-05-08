from __future__ import print_function
import csv
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras import backend as K

#Training Parameters
batch_size = 32
epochs = 12

num_classes = 11

#Model HyperParameters
filter_size= 20
max_pooling_size = 2
dropout_probability = (0.5, 0.9)
hidden_dims = 3

print("Load data...")
ADLTrain = []
with open('S1-ADL1.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S1-ADL2.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S1-ADL3.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S1-ADL4.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S2-ADL1.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S2-ADL2.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S2-ADL3.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S2-ADL4.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S3-ADL1.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S3-ADL2.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S3-ADL3.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)
with open('S3-ADL4.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTrain.append(row)

DrillTrain = []
with open('S1-Drill.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        DrillTrain.append(row)
with open('S2-Drill.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        DrillTrain.append(row)
with open('S3-Drill.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        DrillTrain.append(row)
(x_train, y_train) = ADLTrain, DrillTrain
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

ADLTest = []
with open('S4-ADL1.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTest.append(row)
with open('S4-ADL2.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTest.append(row)
with open('S4-ADL3.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTest.append(row)
with open('S4-ADL4.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTest.append(row)
with open('S4-ADL5.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        ADLTest.append(row)
DrillTest = []
with open('S4-Drill.dat', newline='') as inputfile:
    for row in csv.reader(inputfile):
        DrillTest.append(row)
(x_test, y_test) = ADLTest, DrillTest
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, filter_size, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, filter_size, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(max_pooling_size))




