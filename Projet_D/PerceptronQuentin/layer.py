from random import uniform

from Projet_D.PerceptronQuentin.neuron import *


class Layer:

    def __init__(self, lindex, nbNeu, nbAxonPerNeu, nbDendritesPerNeu, activationFunction): #Axon = output, dendrite = entry of a neuron
        """
        :param nbNeu: without the bias !
        :param nbAxonPerNeu:
        :param nbDendritesPerNeu:
        :param activationFunction:
        """

        self.nbAxonPerNeu = nbAxonPerNeu
        self.nbDendritesPerNeu = nbDendritesPerNeu
        self.nbNeu = nbNeu
        self.neurons = []
        self.activationFunction = activationFunction
        self.lindex = lindex
        for index in range(nbNeu):
            self.neurons.append(Neuron(self.lindex, index, nbDendritesPerNeu, activationFunction))
        self.state = []
        self.deltas = []

    def initEntryLayer(self):
        #On veut : w(0) = [[1],
        #                   ...
        #                  [1]]
        for i in range(self.nbNeu):
            self.neurons[i].setW_b([1], 0)



    def initRandomWeightsAndBias(self):
        #print("nbDendritesPerNeu=", self.nbDendritesPerNeu)
        for i, n in enumerate(self.neurons): #For each neuron of (l)
            neuronW = []
            for j in range(self.nbDendritesPerNeu): #For each neuron of (l-1)
                rand = uniform(0, 0.01)
                neuronW.append(rand)
            b = uniform(0,0.01)
            n.setW_b(neuronW, b)
        #TODO : check the weigh for output (there's no weight)


    def layerPropagation(self, entries):
        self.state = []
        for i, neuron in enumerate(self.neurons): #For each neuron...
            value = neuron.activate(entries)
            self.state.append(value)
        return self.state

    def entryPropagation(self, entries):
        self.state = []
        for i, neuron in enumerate(self.neurons): #For each neuron...
            value = neuron.activate(entries[i])
            self.state.append(value)
        return self.state

    def computeOutputDeltas(self):
        for neuron in self.neurons:
            self.deltas = neuron.computeOutputDelta()
        return self.deltas

    def computeHiddenErrors(self, layer_lplus1):
        for i, neuron in enumerate(self.neurons):
            self.deltas.append(neuron.computeHiddenDelta(layer_lplus1.deltas))
        return self.deltas

    def updateW_b(self, learningRate, layerPartialDerivatives):
        """Partial derivatives must have already been computed"""
        for n, neuron in enumerate(self.neurons):
            neuron.udpateW_b(learningRate, layerPartialDerivatives[n])


    def hadamard(self, l1, l2):
        if len(l1) != len(l2):
            print("Error : wrong Hadamard product")
            exit()
        result = []
        for i in range(len(l1)):
            line1, line2 = l1[i], l2[i]
            resultLine = []
            if len(line1) != len(line2):
                print("Error : wrong Hadamard product")
                exit()
            for j in range(len(line1)):
                resultLine.append(line1[j]*line2[j])
            result.append(resultLine)
        return result

    def learn(self, desiredOutput):
        for n, neu in enumerate(self.neurons):
            neu.updateW_b_old(desiredOutput)

    def printLayer(self):
        print("nbD=",self.nbDendritesPerNeu, "nbNeu=", self.nbNeu, "activation=", self.activationFunction)
        for i, neuron in enumerate(self.neurons):
            neuron.printNeuron()
