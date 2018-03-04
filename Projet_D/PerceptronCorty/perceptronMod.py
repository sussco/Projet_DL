import numpy as np
import random
import math
from copy import deepcopy
import boolDatasets as bd


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
            self.layer.append(np.random.uniform(0, 0.005, size = list_of_layers[i]))
        for j in range(len(list_of_layers)-1):
            self.bias.append(np.random.uniform(0, 0.005, size = list_of_layers[j+1]))
            self.weights.append(np.random.random( (list_of_layers[j+1], list_of_layers[j]) )*0.005)

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
            #print '\n derivees partielles num:',i,' \n'
            #print weights_partial_derivates[-1]
        weights_partial_derivates.reverse()
        bias_partial_derivates = loss_layers


        #print '\n'
        #print bias_partial_derivates
        return(weights_partial_derivates,bias_partial_derivates)
    def backpropagation(self, expected):
        delta_weights = []
        delta_bias = []
        for i in range(len(self.weights)):
            delta_weights.append(np.zeros(shape = np.shape(self.weights[i])))
            delta_bias.append(np.zeros(shape = np.shape(self.bias[i])))
        loss = self.loss_layers(expected)
        for j in range(len(delta_weights)):
            delta_weights[j] += loss[0][j]
            delta_bias[j] += loss[1][j]
        """    print '\n erreur : \n'
        print delta_weights[0]
        print '\n'"""
        return (delta_weights, delta_bias)

    def gradient_descent(self, expected, alpha):
        deltas = self.backpropagation(expected)
        #print "PASSAGE \n",deltas
        for i in range(len(self.weights)):
            #print "PASSAGE n", i, "\n avant: \n"
            #print self.weights[i]
            #print "deltas :\n",alpha*(deltas[0][i]), '\n'
            self.weights[i] = self.weights[i] - alpha*(deltas[0][i] )
            self.bias[i] = self.bias[i] - alpha*(deltas[1][i])
            #print "apres:\n",self.weights[i]

    def multiple_passes(self, inputs, expecteds, alpha):
        assert(len(inputs) == len(expecteds))
        delta_W = []
        delta_B = []
        # initialisation des deltas
        for i in range(len(self.weights)):
            delta_W.append(np.zeros(shape = np.shape(self.weights[i])))
            delta_B.append(np.zeros(shape = np.shape(self.bias[i])))
        print (delta_W)

        for j in range(len(inputs)):
            print (inputs[j])
            print (expecteds[j])
            self.init(inputs[j])
            self.propagation()
            backprop = self.backpropagation(expecteds[j])
            for k in range(len(self.weights)):
                delta_W[k] = delta_W[k] + backprop[0][k]
                delta_B[k] = delta_B[k] + backprop[1][k]
        for l in range(len(self.weights)):
            self.weights[i] -= alpha*((1/len(inputs))*delta_W[i] )
            self.bias[i] -= alpha*((1/len(inputs))*delta_B[i])




    def quadratic_error(self, expected):
        assert(len(self.layer[-1])==len(expected))
        error = 0
        for i in range(len(self.layer[-1])):
            error += (self.layer[-1][i]- expected[i])**2
        return error
