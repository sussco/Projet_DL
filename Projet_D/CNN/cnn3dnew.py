import numpy as np
from random import uniform
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from convLayer import ConvLayer
from  Perceptron import Perceptron
import imageReader

labelled_images = imageReader.list_labelled_images2Dnew('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 15000, 0, 'digits')
test_images = imageReader.list_labelled_images2Dnew('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 5000, 0, 'digits')

nbfilters = 1
conv1 = ConvLayer(nbfilters,28,28,1,3,1,1, 0.5)
conv2 = ConvLayer(nbfilters,28,28,1,3,1,1, 0.5)
fc = Perceptron([784*nbfilters,800,10], 0.5, 0.5)
batch  = 10
count = 0
#conv.filterWeights = np.array([[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]]])
for i in range(int(15000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            activationList = []
            conv1.propagation(labelled_images[0][batch*i+k])


            conv2.propagation(np.array(conv1.activationTable))
            # flatten input for fully connected
            fcInput = np.array(conv2.activationTable).flatten()
            fc.propagationSoftMax(fcInput)
            fc.backPropagationCE(labelled_images[1][batch*i+k])
            #print(labelled_images[0][batch*i+k].shape)
            deltaTable = np.reshape(fc.lossPerLayer[0], (1, conv2.entryD, conv2.entryW, conv2.entryH))
            conv2.computeWeightsTable(conv1.activationTable, deltaTable)
            conv2.computeDeltaTable(fc.lossPerLayer[0])
            deltaTable2 = np.reshape(conv2.deltaTable, (1,1,28,28))
            conv1.computeWeightsTable(labelled_images[0][batch*i+k], deltaTable2)

        #print(conv.filterWeights)
        #print(conv.filterWeightsTable)
        fc.updateParams(batch)
        conv2.updateParams(batch)
        conv1.updateParams(batch)
        print (i)
        if (np.argmax(fc.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1

count_test = 0
for j in range(5000):
    activationListTest = []
    conv1.propagation(test_images[0][j])
    conv2.propagation(np.array(conv1.activationTable))
    fcInput = np.array(conv2.activationTable).flatten()
    fc.propagationSoftMax(fcInput)
    if (np.argmax(fc.layers[-1]) == np.argmax(test_images[1][j])):
        count_test +=1
    print(count_test/float(j+1))
