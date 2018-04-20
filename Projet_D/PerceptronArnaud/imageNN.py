import Perceptron as per
import inmageTest
import numpy as np
import random
import math
import pickle
#print [1-0.9514, 1-0.9539, 1-0.9509, 1-0.9536, 1-0.9514]

labelled_images = inmageTest.list_labelled_images('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 30000, 0, 'digits')
test_images = inmageTest.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0, 'digits')
mean_list = []
nb = 800
lay = [784,nb,10]
learningR = 0.1
batch = 1
fRes= open("results24","ab")
for o in range(15):
    mean_list.append(lay)
    mean_list.append(learningR)
    mean_list.append(batch)
    percep = per.Perceptron(lay, learningR, 0)
    count = 0
        #print percep.layer[1], '\n \n'
    for i in range(int(30000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            percep.propagationSoftMax_ReLU(labelled_images[0][batch*i+k])
            percep.backPropagationCE_RELU(labelled_images[1][batch*i+k])
        percep.updateParams(batch)
        print (i)
        if (np.argmax(percep.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        print( count/float(i+1))
    count_test = 0
    for j in range(10000):
        percep.propagationSoftMax_ReLU(test_images[0][j])

        if (np.argmax(percep.layers[-1]) == np.argmax(test_images[1][j])):
            count_test +=1
        print (count_test/float(j+1))
        print (percep.layers[-1])
        print (test_images[1][j])
    mean_list.append(count_test/float(j+1))
    learningR += 0.1
print (mean_list)
pickle.dump(mean_list,fRes)
