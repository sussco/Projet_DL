import numpy
from Perceptron import *
from Relu import *
from Pool import *
from convLayer import *
from Sigmoid import *

class Net():

    def __init__(self):
        conv1 = ConvLayer(28, 28, 1, 6, 5, 1, 2)
        sigmoid1 = Sigmoid()
        pool1 = Pool(2)
        conv2 = ConvLayer(14, 14, 6, 16, 5, 1, 0)
        sigmoid2 = Sigmoid()
        pool2 = Pool(2)
        fc = Perceptron([400, 700, 26])

        self.layers = [conv1, sigmoid1, pool1, conv2, sigmoid2, pool2, fc]


    def propagation(self, input):
        inputLay = input
        for lay in self.layers:
            # print(lay)
            inputLay = lay.propagation(inputLay)
            # print(inputLay.shape)
            # print(inputLay.shape)

        # print (inputLay)
        return inputLay

    def backPropagation(self, outPut):
        outputLay = outPut
        for lay in self.layers[::-1]:
            outputLay = lay.backPropagation(outputLay)
            # print(lay)
            # print(outputLay.shape)
            #print(outputLay[0])


    def updateParams(self, batchSize, learningR):
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, Perceptron)):
                layer.updateParams(batchSize, learningR)


    def train(self, inputs, labels, batchSize, learningR):
        count_test = 0
        for i in range(int(len(inputs)/batchSize)):
            for k in range(batchSize):
                self.propagation(inputs[batchSize*i+k])
                self.backPropagation(labels[batchSize*i+k])
            if (np.argmax(self.propagation(inputs[batchSize*i+k])) == np.argmax(labels[batchSize*i+k])):
                count_test +=1
            if(i%100 == 0):
                print(count_test/float(i+1))
            self.updateParams(batchSize, learningR)
            #print(self.layers[0].activationTable)
            # print(self.layers[0].filterTable[0][])
        print(count_test/float(i+1))


    def test(self, inputs, labels):
        count_test = 0
        resPerClass = np.zeros(26)
        for i in range(len(inputs)):
                if (np.argmax(self.propagation(inputs[i])) == np.argmax(labels[i])):
                    resPerClass[np.argmax(labels[i])] += 1
                    count_test +=1
                # print(count_test/float(i+1))
        print(resPerClass/1000.0)
        print(count_test/float(i+1))
        return count_test/float(i+1)
        #print(self.layers[0].filterTable)
        # print(self.layers[3].filterTable)
