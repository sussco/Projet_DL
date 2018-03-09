#import Layer
import numpy as np
import math



def sigmoid(x):
    try:
        ans = 1 / (1 + math.exp(-x))
    except OverflowError:
        ans = float('inf')
    return ans

def vector_sigmoid(x):
    for i in range(len(x)):
        x[i] = sigmoid(x[i])
    return x

def softmax(x):
    sum = 0.0
    for i in range(len(x)):
        sum += math.exp(x[i])
    return np.exp(x)/sum


class Perceptron:


    def __init__(self, list_of_layers, learningRate, importance):
        self.weightsTable = []
        self.biaisTable = []
        self.layers = []
        self.biais = []
        self.weights = []
        for i in range(len(list_of_layers)):
            self.layers.append(np.random.uniform(0, 0.01, size = list_of_layers[i]))
        for j in range(len(list_of_layers)-1):
            self.biais.append(np.random.uniform(0, 0.01, size = list_of_layers[j+1]))
            self.weights.append(np.random.random((list_of_layers[j+1], list_of_layers[j]) )*0.01)
            self.weightsTable.append(np.zeros([list_of_layers[j+1], list_of_layers[j]]))
            self.biaisTable.append(np.zeros([list_of_layers[j + 1]]))
        self.learningRate = learningRate
        self.importance = importance




    def propagation(self, layIn):
        assert len(layIn) == len(self.layers[0])
        self.layers[0] = np.array(layIn)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1] = vector_sigmoid(np.matmul(self.weights[i],self.layers[i]) + self.biais[i])
        return self.layers[-1]

    def propagationSoftMax(self, layIn):
        assert len(layIn) == len(self.layers[0])
        self.layers[0] = np.array(layIn)
        for i in range(len(self.layers) -2):
            self.layers[i + 1] = vector_sigmoid(np.matmul(self.weights[i],self.layers[i]) + self.biais[i])
        self.layers[len(self.layers)-1] = softmax(np.matmul(self.weights[len(self.layers)-2],self.layers[len(self.layers)-2]) + self.biais[len(self.layers)-2])

    def backPropagation(self, expectedOutput):
        lossPerLayer = []
        lossPerLayer.append(-(expectedOutput - self.layers[-1]) * (self.layers[-1]*(1 - self.layers[-1])))
        for l in range(len(self.layers) - 2, -1, -1):
            lossPerLayer.append(np.matmul(np.transpose(self.weights[l]),lossPerLayer[-1])*(self.layers[l] * (1 - self.layers[l])))
        lossPerLayer.reverse()
        for l in range(len(lossPerLayer) - 1):
            self.weightsTable[l] = np.outer(lossPerLayer[l + 1], np.transpose(self.layers[l]))
            self.biaisTable[l] = lossPerLayer[l + 1]

    def backPropagationCE(self, expectedOutput):
        lossPerLayer = []
        lossPerLayer.append(-(expectedOutput - self.layers[-1]))
        for l in range(len(self.layers) - 2, -1, -1):
            lossPerLayer.append(np.matmul(np.transpose(self.weights[l]),lossPerLayer[-1])*(self.layers[l] * (1 - self.layers[l])))
        lossPerLayer.reverse()
        for l in range(len(lossPerLayer) - 1):
            self.weightsTable[l] = np.outer(lossPerLayer[l + 1], np.transpose(self.layers[l]))
            self.biaisTable[l] = lossPerLayer[l + 1]


    def updateParams(self, nbTrainings):
        for l in range(len(self.layers) - 1):
            self.weights[l] -= self.learningRate * ( 1/float(nbTrainings) * self.weightsTable[l])
            self.biais[l] -= self.learningRate * ( 1/float(nbTrainings) * self.biaisTable[l])

    def quadratic_error(self, expected):
        error = 0
        for i in range(len(self.layers[-1])):
            error += (self.layers[-1][i]- expected[i])**2
        return error




#
# test = Perceptron([10,20,10,10],1,0)
#
# for l in range(6 - 2, 0 ,-1):
#     print(l)
# for i in range(10000):
#     t = [np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),
#     np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2)]
#     print(t)
#     print(test.propagation(t))
#     test.backPropagation([ (1-t[i]) for i in range(10)])
#     test.updateParams(1)

# print('\n')
# print(test.propagation([1,1,1,1]))
# print(test.propagation([0,0,1,1]))
# print(test.propagation([1,1,0,0]))
# print(test.propagation([0,1,0,1]))
# print(test.propagation([0,0,0,0]))
# print(test.propagation([0,1,1,0]))
# print(test.propagation([1,0,0,1]))
# print(test.propagation([1,0,1,1]))
