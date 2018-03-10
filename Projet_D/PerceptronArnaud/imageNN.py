import Perceptron as per
import inmageTest
import numpy as np
import random
import math
import pickle
#print [1-0.9514, 1-0.9539, 1-0.9509, 1-0.9536, 1-0.9514]

labelled_images = inmageTest.list_labelled_images('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 60, 0, 'digits')
test_images = inmageTest.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10, 0, 'digits')
mean_list = []
lay = [784,800,10]
learningR = 0.05

fRes= open("test1","ab")
for o in range(1):
    mean_list.append(lay)
    mean_list.append(learningR)
    percep = per.Perceptron(lay, learningR, 0)
    count = 0
        #print percep.layer[1], '\n \n'
    for i in range(60):
        #print percep.layer[1], '\n \n'
        percep.propagationSoftMax(labelled_images[0][i])
        percep.backPropagationCE(labelled_images[1][i])
        percep.updateParams(1)
        if (np.argmax(percep.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        print count/float(i+1)
    mean_list.append(count/float(i+1))
    count_test = 0
    for j in range(10):
        percep.propagationSoftMax(test_images[0][j])

        if (np.argmax(percep.layers[-1]) == np.argmax(test_images[1][j])):
            count_test +=1
        print count_test/float(j+1)
        print percep.layers[-1]
        print test_images[1][j]
    print "ajout"
    mean_list.append(count_test/float(j+1))
    pickle.dump(mean_list,fRes)

print mean_list

"""print percep.quadratic_error(labelled_images[1][i])
    print percep.layer[-1] , '\n'
    print labelled_images[1][i] , '\n'"""
"""
count = 0
for i in range(30000,59999):
    percep.init(labelled_images[0][i])
    #print percep.layer[1], '\n \n'
    percep.propagation()
    if (np.argmax(percep.layer[-1]) == np.argmax(labelled_images[1][i])):
        count +=1
    print count/float(i-29999)"""
