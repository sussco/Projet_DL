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
        try:
            sum += math.exp(x[i])
        except OverflowError:
            sum += float('inf')
    return np.exp(x)/sum

def reLU(x):
    if(x>0):
        return x
    else:
        return 0.01*x

def vector_reLU(x):
    for i in range(len(x)):
        x[i] = reLU(x[i])
    return x

def backProp_ReLU(x):
    back = []
    for i in range(len(x)):
        back.append(1) if x[i]>0 else back.append(0.01)
    return back


class Perceptron:


    def __init__(self, list_of_layers):
        self.weightsTable = []
        self.biaisTable = []
        self.layers = []
        self.biais = []
        self.weights = []
        self.lossPerLayer = []

        for i in range(len(list_of_layers)):
            self.layers.append(np.random.uniform(0, 0.05, size = list_of_layers[i]))
        for j in range(len(list_of_layers)-1):
            self.biais.append(np.random.uniform(0, 0.05, size = list_of_layers[j+1]))
            self.weights.append(np.random.random((list_of_layers[j+1], list_of_layers[j]) )*0.05)
            self.weightsTable.append(np.zeros([list_of_layers[j+1], list_of_layers[j]]))
            self.biaisTable.append(np.zeros([list_of_layers[j + 1]]))




    def propagation_Normal(self, layIn):
        self.inShape = layIn.shape
        layIn = np.array(layIn).flatten()
        print(len(layIn))
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

    def propagation(self, layIn):
        self.inShape = layIn.shape
        layIn = np.array(layIn).flatten()
        assert len(layIn) == len(self.layers[0])
        self.layers[0] = np.array(layIn)
        for i in range(len(self.layers) -2):
            self.layers[i + 1] = vector_reLU(np.matmul(self.weights[i],self.layers[i]) + self.biais[i])
        self.layers[len(self.layers)-1] = softmax(np.matmul(self.weights[len(self.layers)-2],self.layers[len(self.layers)-2]) + self.biais[len(self.layers)-2])


    def backPropagation_Normal(self, expectedOutput):
        self.lossPerLayer = []
        self.lossPerLayer.append(-(expectedOutput - self.layers[-1]) * (self.layers[-1]*(1 - self.layers[-1])))
        for l in range(len(self.layers) - 2, -1, -1):
            self.lossPerLayer.append(np.matmul(np.transpose(self.weights[l]),self.lossPerLayer[-1])*(self.layers[l] * (1 - self.layers[l])))
        self.lossPerLayer.reverse()
        for l in range(len(self.lossPerLayer) - 1):
            self.weightsTable[l] += np.outer(self.lossPerLayer[l + 1], np.transpose(self.layers[l]))
            self.biaisTable[l] += self.lossPerLayer[l + 1]
        return np.reshape(self.lossPerLayer[0], self.inShape)

    def backPropagation(self, expectedOutput):
        self.lossPerLayer = []
        self.lossPerLayer.append(-(expectedOutput - self.layers[-1]))
        for l in range(len(self.layers) - 2, -1, -1):
            a = np.matmul(np.transpose(self.weights[l]),self.lossPerLayer[-1])*(self.layers[l] * (1 - self.layers[l]))
            self.lossPerLayer.append(a)
        self.lossPerLayer.reverse()
        for l in range(len(self.lossPerLayer) - 1):
            self.weightsTable[l] += np.outer(self.lossPerLayer[l + 1], np.transpose(self.layers[l]))
            self.biaisTable[l] += self.lossPerLayer[l + 1]
        return np.reshape(self.lossPerLayer[0], self.inShape)

    def backPropagationCE_RELU(self, expectedOutput):
        self.lossPerLayer = []
        self.lossPerLayer.append(-(expectedOutput - self.layers[-1]))
        for l in range(len(self.layers) - 2, -1, -1):
            a = np.matmul(np.transpose(self.weights[l]),self.lossPerLayer[-1])*backProp_ReLU(self.layers[l])
            self.lossPerLayer.append(a)
        self.lossPerLayer.reverse()
        for l in range(len(self.lossPerLayer) - 1):
            self.weightsTable[l] += np.outer(self.lossPerLayer[l + 1], np.transpose(self.layers[l]))
            self.biaisTable[l] += self.lossPerLayer[l + 1]


    def updateParams(self, nbTrainings, learningR):
        for l in range(len(self.layers) - 1):
            self.weights[l] -= learningR * ( 1/float(nbTrainings) * self.weightsTable[l])
            self.weightsTable[l] = 0
            self.biais[l] -= learningR * ( 1/float(nbTrainings) * self.biaisTable[l])
            self.biaisTable[l] = 0

    def quadratic_error(self, expected):
        error = 0
        for i in range(len(self.layers[-1])):
            error += (self.layers[-1][i]- expected[i])**2
        return error
