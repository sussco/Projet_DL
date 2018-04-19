import sys
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
conv1 = ConvLayer(nbfilters,28,28,1,3,1,0, 0.7)
conv2 = ConvLayer(nbfilters,26,26,1,3,1,0, 0.7)
fc = Perceptron([24*24*nbfilters,800,10], 0.7, 0.7    )
batch  = 10
count = 0
b = deepcopy(conv2.filterTable)
a = deepcopy(conv1.filterTable)
for i in range(int(5000/int(batch))):
        # print("CONV1: ",conv1.filterTable)
        # print("CONV2: ",conv2.filterTable)
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            conv1.propagation(labelled_images[0][batch*i+k])

            conv2.propagation(np.array(conv1.activationTable))
            # flatten input for fully connected
            fcInput = np.array(conv2.activationTable).flatten()
            fc.propagationSoftMax(fcInput)
            fc.backPropagationCE(labelled_images[1][batch*i+k])
            #print(labelled_images[0][batch*i+k].shape)
            # print("conv2 entries : ", conv2.entryW, conv2.entryH)
            deltaTable = np.reshape(fc.lossPerLayer[0], (1, conv2.entryD, conv2.layW, conv2.layH))
            #print("DETLA FC : ", deltaTable)
            # print("deltaTable BEFORE: ", conv2.deltaTable[0][0,0])
            conv2.computeDeltaTable(deltaTable, conv1.activationTable)
            # print("deltaTable AFTER: ", conv2.deltaTable[0][0,0])

            #print(conv2.deltaTable)
            conv2.computeWeightsTable(deltaTable)
            # print("fc layer : ", len(fc.lossPerLayer[0]))
            # print("deltaTable 2: ", np.array(conv2.deltaTable).shape)
            deltaTable2 = np.reshape(np.array(conv2.deltaTable), (1,1,26,26))
            conv1.computeWeightsTable(deltaTable2)
            #print(fc.layers[-1])
            # print("ENTREE: ", np.reshape(np.array(fc.layers[0]), (24,24))[15])
            # print("LABEL: ", np.argmax(labelled_images[1][batch*i+k]))
        #print(conv.filterWeightsTable)
        fc.updateParams(batch)
        conv2.updateParams(batch)
        conv1.updateParams(batch)
        # print(conv1.filterTable)
        # print(conv1.filterTable)
        print (i)
        if (np.argmax(fc.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1

count_test = 0
for j in range(5000):
    conv1.propagation(test_images[0][j])
    conv2.propagation(np.array(conv1.activationTable))
    fcInput = np.array(conv2.activationTable).flatten()
    fc.propagationSoftMax(fcInput)
    print(fc.layers[-1])
    if (np.argmax(fc.layers[-1]) == np.argmax(test_images[1][j])):
        count_test +=1
    print(count_test/float(j+1))
    print(np.argmax(test_images[1][j]))
    print("CONV1: ", np.array(conv1.activationTable)[0,0,12])
    print("CONV2: ", np.array(conv2.activationTable)[0,0,12])
    print(fc.layers[-1])
    #plt.matshow(np.reshape(np.array(conv1.activationTable), (26,26)),cmap=plt.cm.gray)
print("BEGINNING: ", a, b)
print("END: ", conv1.filterTable, conv2.filterTable)
#plt.show()
