import Perceptron as per
import imageReader
import numpy as np
import random
import math
import pickle
#print [1-0.9514, 1-0.9539, 1-0.9509, 1-0.9536, 1-0.9514]

labelled_images = imageReader.list_labelled_images('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 60000, 0, 'digits')
test_images = imageReader.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0, 'digits')
mean_list = []
nb = 800
lay = [784, 400, 120, 84, 10]
learningR = 0.5
batch = 10
fRes= open("results24","ab")
for o in range(1):
    mean_list.append(lay)
    mean_list.append(learningR)
    mean_list.append(batch)
    percep = per.Perceptron(lay)
    count = 0
        #print percep.layer[1], '\n \n'
    for i in range(int(60000/int(batch))):
        #print percep.layer[1], '\n \n'
        for k in range(batch):
            percep.propagation(labelled_images[0][batch*i+k])
            percep.backPropagation(labelled_images[1][batch*i+k])
            print(percep.layers[-1])
        percep.updateParams(batch, learningR)
        print (count/float(i + 1))
        if (np.argmax(percep.layers[-1]) == np.argmax(labelled_images[1][i])):
            count +=1
        print( i+1)
    count_test = 0
    for j in range(10000):
        percep.propagation(test_images[0][j])

        if (np.argmax(percep.layers[-1]) == np.argmax(test_images[1][j])):
            count_test +=1
        print (count_test/float(j+1))
        print (percep.layers[-1])
        print (test_images[1][j])
    mean_list.append(count_test/float(j+1))
    learningR += 0.1
print (mean_list)
pickle.dump(mean_list,fRes)
