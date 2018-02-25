import numpy as np
import random
import math
from copy import deepcopy



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def vector_sigmoid(x):
    for i in range(len(x)):
        x[i] = sigmoid(x[i])
    return x


class perceptron:

    def __init__(self, list_of_layers):
        self.layer = []
        self.bias = []
        self.weights = []
        for i in range(len(list_of_layers)):
            self.layer.append(np.random.uniform(0, 0.05, size = list_of_layers[i]))
        for j in range(len(list_of_layers)-1):
            self.bias.append(np.random.uniform(0, 0.05, size = list_of_layers[j+1]))
            self.weights.append(np.random.random( (list_of_layers[j+1], list_of_layers[j]) )*0.05)

    def propagation(self):
        for i in range(len(self.layer)-1):
            self.layer[i+1] = vector_sigmoid(np.matmul(self.weights[i], self.layer[i]) + self.bias[i])

    def init(self,input_object):
        assert len(input_object) == len(self.layer[0])
        self.layer[0] = np.array(input_object)

    def loss_layers(self, expected):
        loss_layers = []

        loss_layers.append(-(expected - self.layer[-1])*(self.layer[-1]*(1-self.layer[-1])) )
        for i in range(len(self.layer)-2):
            """ attention self.layers est plus grande que self.weights de 1"""
            loss_layers.append(np.matmul(np.transpose(self.weights[-(i+1)]), loss_layers[-1])*(self.layer[-(i+2)]*(1-self.layer[-(i+2)])))
        loss_layers.reverse()
        weights_partial_derivates = []
        bias_partial_derivates = []
        for i in range(len(loss_layers)):
            weights_partial_derivates.append(np.outer(loss_layers[-(i+1)], self.layer[-(i+2)]))
        weights_partial_derivates.reverse()
        bias_partial_derivates = loss_layers

        """print '\n derivees partielles: \n'
        print weights_partial_derivates
        print '\n'
        print bias_partial_derivates"""
        return(weights_partial_derivates,bias_partial_derivates)

    def backpropagation(self, expecteds):
        delta_weights = []
        delta_bias = []
        for i in range(len(self.weights)):
            delta_weights.append(np.zeros(shape = np.shape(self.weights[i])))
            delta_bias.append(np.zeros(shape = np.shape(self.bias[i])))
        for i in range(len(expecteds)):
            loss = self.loss_layers(expecteds[i])
            for j in range(len(delta_weights)):
                delta_weights[j] += loss[0][j]
                delta_bias[j] += loss[1][j]
        """    print '\n erreur : \n'
        print delta_weights[0]
        print '\n'
        print delta_weights[1]"""
        return (delta_weights, delta_bias)

    def gradient_descent(self, expecteds, alpha):
        deltas = self.backpropagation(expecteds)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - alpha*((1/len(expecteds)*deltas[0][i] ))
            self.bias[i] = self.bias[i] - alpha*((1/len(expecteds)*deltas[1][i] ))

    def quadratic_error(self, expected):
        error = 0
        for i in range(len(self.layer[-1])):
            error += (self.layer[-1][i]- expected[i])**2
        return error


def gen_dataset(length):
    data = []
    labels = []
    for i in range(length):
        in1 = random.randint(0,1)
        in2 = random.randint(0,1)
        data.append([in1, in2])
        labels.append([in1*in2])
    return (data,labels)



a = perceptron([2,2,1])
a.init([1,0])
a.propagation()
print '\n'
print 'valeur des neuronnes au debut:\n', a.layer, '\n'

dataset = gen_dataset(1000)
for i in range(1000):
    a.init(dataset[0][i])
    a.propagation()
    a.gradient_descent(dataset[1][i], 2)
    print a.quadratic_error(dataset[1][i])
