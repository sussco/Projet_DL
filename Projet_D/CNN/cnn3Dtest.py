import numpy as np
from random import uniform
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from convlayer import ConvLayer
from  Perceptron import Perceptron
import imageReader

labelled_images = imageReader.list_labelled_images2D('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 15000, 0, 'digits')
test_images = imageReader.list_labelled_images2D('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 1000, 0, 'digits')

nbfilters = 2
conv = ConvLayer(nbfilters,28,28,1,3,1,1, 0.7)
fc = Perceptron([784*nbfilters,800,10], 0.7, 1)
batch  = 10
count = 0
#conv.filterWeights = np.array([[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]]])
for i in range(int(15000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            activationList = []
            conv.propagation(labelled_images[0][batch*i+k])
            # flatten input for fully connected
            for m in range(nbfilters):
                activationList.append(conv.conv2Dlayers[m].activationTable)

            fcInput = np.array(activationList).flatten()
            #print(fcInput.shape)
            fc.propagationSoftMax(fcInput)
            fc.backPropagationCE(labelled_images[1][batch*i+k])
            #print(labelled_images[0][batch*i+k].shape)
            conv.computeWeightsTable(labelled_images[0][batch*i+k], np.reshape(fc.lossPerLayer[0], (28,28,2)))
        #print(conv.filterWeights)
        #print(conv.filterWeightsTable)
        fc.updateParams(batch)
        conv.updateParams(batch)
        print (i)
        if (np.argmax(fc.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        #print(conv.filterWeights)

count_test = 0
for j in range(1000):
    activationListTest = []
    conv.propagation(test_images[0][j])
    for m in range(nbfilters):
        activationListTest.append(conv.conv2Dlayers[m].activationTable)

    fcInput = np.array(activationListTest).flatten()
    fc.propagationSoftMax(fcInput)
    if (np.argmax(fc.layers[-1]) == np.argmax(test_images[1][j])):
        count_test +=1
    print(count_test/float(j+1))
