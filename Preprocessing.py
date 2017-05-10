import cPickle as cp
import numpy as np
import keras
from keras.preprocessing import image
SLIDING_WINDOW_LENGTH = 64
SLIDING_WINDOW_OVERLAP = 32

OPPORTUNITY_FILE_NAMES = ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S2-Drill.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S3-Drill.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat']

#keras.preprocessing.image.ImageDataGenerator(samplewise_std_normalization=True)

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def load_dataset(filename):
    f = file(filename, 'rb')
    #serializing data
    data = cp.load(f)
    f.close()

    x_train, y_train = data[0]
    x_test, y_test = data[1]

    return x_train, y_train, x_test, y_test

print "loading data..."
for fileName in OPPORTUNITY_FILE_NAMES:
    x_train, y_train, x_test, y_test = load_dataset(fileName)
