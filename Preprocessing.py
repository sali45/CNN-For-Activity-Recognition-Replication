import cPickle as cp
import numpy as np
from itertools import islice
import keras
from keras.preprocessing import image
SLIDING_WINDOW_LENGTH = 64
SLIDING_WINDOW_OVERLAP = 32

OPPORTUNITY_FILE_NAMES = ['OpportunityChallengeLabeled/S1-Drill.dat',
                          'OpportunityChallengeLabeled/S1-ADL1.dat',
                          'OpportunityChallengeLabeled/S1-ADL2.dat',
                          'OpportunityChallengeLabeled/S1-ADL3.dat',
                          'OpportunityChallengeLabeled/S1-ADL4.dat',
                          'OpportunityChallengeLabeled/S1-ADL5.dat',
                          'OpportunityChallengeLabeled/S2-Drill.dat',
                          'OpportunityChallengeLabeled/S2-ADL1.dat',
                          'OpportunityChallengeLabeled/S2-ADL2.dat',
                          'OpportunityChallengeLabeled/S2-ADL3.dat',
                          'OpportunityChallengeLabeled/S2-ADL4.dat',
                          'OpportunityChallengeLabeled/S2-ADL5.dat',
                          'OpportunityChallengeLabeled/S3-Drill.dat',
                          'OpportunityChallengeLabeled/S3-ADL1.dat',
                          'OpportunityChallengeLabeled/S3-ADL2.dat',
                          'OpportunityChallengeLabeled/S3-ADL3.dat',
                          'OpportunityChallengeLabeled/S3-ADL4.dat',
                          'OpportunityChallengeLabeled/S3-ADL5.dat',
                          'OpportunityChallengeLabeled/S4-Drill.dat',
                          'OpportunityChallengeLabeled/S4-ADL1.dat',
                          'OpportunityChallengeLabeled/S4-ADL2.dat',
                          'OpportunityChallengeLabeled/S4-ADL3.dat',
                          'OpportunityChallengeLabeled/S4-ADL4.dat',
                          'OpportunityChallengeLabeled/S4-ADL5.dat']

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def load_dataset(filename):
    f = file(filename, 'rb')
    #serializing data
    data = cp.load(f)
    f.close()

    feature_normalize(data)
    x_train, y_train = window(data)[0]
    x_test, y_test = window(data)[1]

    return x_train, y_train, x_test, y_test

def window(seq, n=SLIDING_WINDOW_OVERLAP):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

print "loading data..."
for fileName in OPPORTUNITY_FILE_NAMES:
    x_train, y_train, x_test, y_test = load_dataset(fileName)
