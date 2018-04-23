import numpy
from Perceptron import *
from Relu import *
from Pool import *
from convLayer import *

class Net():

    def __init__(self):

        conv1 = ConvLayer(28, 28, 1, 6, 5, 1, 2)
        relu1 = ReLU()
        pool1 = Pool(2)
        conv2 = ConvLayer(14, 14, 6, 16, 5, 1, 0)
        relu2 = ReLU()
        pool2 = Pool(2)
        fc = Perceptron([400, 120, 84, 10])

        self.layers = [conv1, relu1, pool1, conv2, relu2, pool2, fc]


    def propagation(self, input):
        inputLay = input
        for lay in self.layers:
            inputLay = lay.propagation(inputLay)
        print(inputLay)
        return inputLay

    def backPropagation(self, outPut):
        outputLay = outPut
        for lay in self.layers[::-1]:
            outputLay = lay.backPropagation(outputLay)

    def updateParams(self, batchSize, learningR):
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, Perceptron)):
                layer.updateParams(batchSize, learningR)

    def train(self, inputs, labels, batchSize, learningR):
        for i in range(int(len(inputs)/batchSize)):
            for k in range(batchSize):
                self.propagation(inputs[batchSize*i+k])
                self.backPropagation(labels[batchSize*i+k])
            print(i)
            self.updateParams(batchSize, learningR)

    def test(self, inputs, labels):
        count_test = 0
        for i in range(len(inputs)):
                output = self.propagation(inputs[i])
                if (np.argmax(output) == np.argmax(labels[i])):
                    count_test +=1
                print(count_test/float(i+1))
