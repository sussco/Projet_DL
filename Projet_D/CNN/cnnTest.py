import numpy as np
from random import uniform
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from convlayer import ConvLayer2D
from  Perceptron import Perceptron
import imageReader

labelled_images = imageReader.list_labelled_images2D('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 15000, 0, 'digits')
test_images = imageReader.list_labelled_images2D('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0, 'digits')

conv = ConvLayer2D(28,28,1,3,1,1, 0.7)
fc = Perceptron([784,500,10], 0.7, 1)
batch  = 10
count = 0
#conv.filterWeights = np.array([[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]]])
for i in range(int(15000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            conv.feedforward(labelled_images[0][batch*i+k])
            # flatten input for fully connected
            fcInput = np.reshape(conv.activationTable, (784))
            fc.propagationSoftMax(fcInput)
            fc.backPropagationCE(labelled_images[1][batch*i+k])
            conv.computeWeightsTable(labelled_images[0][batch*i+k], np.reshape(fc.lossPerLayer[0], (28,28,1)))
        #print(conv.filterWeights)
        #print(conv.filterWeightsTable)
        fc.updateParams(batch)
        conv.updateParams(batch)
        print (i)
        if (np.argmax(fc.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        print(conv.filterWeights)

count_test = 0
for j in range(10000):
    conv.feedforward(test_images[0][j])
    fcInput = np.reshape(conv.activationTable, (784))
    fc.propagationSoftMax(fcInput)
    if (np.argmax(fc.layers[-1]) == np.argmax(test_images[1][j])):
        count_test +=1
        print(count_test/float(j+1))
