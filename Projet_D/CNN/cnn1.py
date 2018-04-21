import sys
import numpy as np
from random import uniform
from random import shuffle
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from convLayer import ConvLayer
from  Perceptron import Perceptron
import imageReader

labelled_images = imageReader.list_labelled_images2Dnew('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 30000, 0, 'digits')
test_images = imageReader.list_labelled_images2Dnew('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0, 'digits')
shuffle(labelled_images)
nbfilters = 1
learningR = 0.1
conv1 = ConvLayer(nbfilters,28,28,1,3,1,0, learningR)
fc = Perceptron([26*26*nbfilters,500,10], learningR, learningR)
batch  = 10
count = 0
a = deepcopy(conv1.filterTable)

#conv.filterWeights = np.array([[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[0]]])
for i in range(int(15000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            conv1.propagation_ReLU(labelled_images[batch*i+k][0])
            # flatten input for fully connected
            fcInput = np.array(conv1.activationTable).flatten()
            fc.propagationSoftMax_ReLU(fcInput)
            fc.backPropagationCE_RELU(labelled_images[batch*i+k][1])
            # print(len(fc.lossPerLayer[0]))
            deltaTable = np.reshape(fc.lossPerLayer[0], (nbfilters, conv1.entryD, conv1.layW, conv1.layH))
            conv1.computeWeightsTable_Test(deltaTable)
            #print(fc.layers[-1])
        #print(conv.filterWeights)
        print(conv1.filterTable)
        print(fc.layers[-1])
        fc.updateParams(batch)
        conv1.updateParams(batch)
        print (i)
        if (np.argmax(fc.layers[-1]) == np.argmax(labelled_images[i][1])):
            count +=1
        # print(fc.layers[-1])
        # print(labelled_images[i][1])
        # print(count/float(i+1))
count_test = 0
for j in range(10000):
    conv1.propagation_ReLU(test_images[j][0])
    fcInput = np.array(conv1.activationTable).flatten()
    fc.propagationSoftMax_ReLU(fcInput)
    #print("ENTREE: ", np.reshape(np.array(fc.layers[0]), (26,26))[15])
    if (np.argmax(fc.layers[-1]) == np.argmax(test_images[j][1])):
        count_test +=1
    print(count_test/float(j+1))
    print(np.argmax(fc.layers[-1]))
    print(np.argmax(test_images[j][1]))
    # print("INPUT",  np.reshape(np.array(test_images[j][0]), (28,28))[0:3,0:3])
    # print("CONV OUT", conv1.activationTable[0][0,0,0])
    print(fc.layers[-1])
print(learningR)
"""for lk in range(nbfilters):
    print(conv1.filterTable[lk].shape)
    plt.matshow(np.reshape(conv1.filterTable[lk], (3,3)) ,cmap=plt.cm.gray)"""

#print("BEGINNING: ", a)
print("END: ", conv1.filterTable)
#print(conv1.biasTable)
# for i in range(nbfilters):
#     plt.matshow(conv1.filterTable[i][0], cmap = 'gray')
# plt.show()
