import perceptronMod as per
import inmageTest
import numpy as np
import random
import math

labelled_images = inmageTest.list_labelled_images('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 60000, 0, 'digits')
"""test_images = inmageTest.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0)
"""
#labelled_letters = inmageTest.list_labelled_images('emnist-letters-train-images-idx3-ubyte', 'emnist-letters-train-labels-idx1-ubyte', 60000, 0, 'letters')

mean_list = []
for o in range(1):
    percep = per.perceptron([784,300, 10])
    count = 0
    for i in range(60000):
        percep.init(labelled_images[0][i])
        #print percep.layer[1], '\n \n'
        percep.propagation()
        percep.gradient_descent(labelled_images[1][i], 0.6)
        if (np.argmax(percep.layer[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        print (count/float(i+1))
        print (percep.layer[-1])
        print (labelled_images[1][i])

    """count_test = 0
    for j in range(10000):
        percep.init(test_images[0][j])
        #print percep.layer[1], '\n \n'
        percep.propagation()
        if (np.argmax(percep.layer[-1]) == np.argmax(test_images[1][j])):
            count_test +=1
        print (count_test/float(j+1))
        print (percep.layer[-1])
        print (test_images[1][i])
    print ("ajout")
    mean_list.append(count_test/float(j+1))

print (mean_list)"""

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
