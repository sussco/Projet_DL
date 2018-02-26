from random import uniform
import math
import numpy as np
from layer import *

class Perceptron:
    """Neuronal Network, defined by nb of layers, nbEntries, nbHypotheses and n/layer"""

    def __init__(self, nbHiddenLayers, nbNeuronsPerLayer, nbHypotheses, nbInputs, learningRate, relativeImportance, activationFunction):
        """Creating a perceptron:

            nbAxonPerNeu = nb outputs for 1 neuron
            nbDendritePerNeu = nb inputs per neuron

            For a layer, all neurons are connected to all neurons of previous and next layer.
        """

        super().__init__()
        self.nbHiddenLayers = nbHiddenLayers
        self.nbNeuronsPerLayer = nbNeuronsPerLayer
        self.nbHypotheses = nbHypotheses
        self.nbInputs = nbInputs
        self.learningRate = learningRate
        self.relativeImportance = relativeImportance
        self.nbLayers = nbHiddenLayers + 2
        self.layers = []

        #Entry layer
        if nbHiddenLayers <= 0:
            self.layers.append(Layer(0, nbInputs, nbHypotheses, 1, "entry"))
        else:
            self.layers.append(Layer(0, nbInputs, nbNeuronsPerLayer, 1, "entry"))
        #Hidden layers
        for i in range(nbHiddenLayers):
            if i == 0: #Première couche cachée, attention au nombre d'entrées
                if nbHiddenLayers == 1: #..if there's only one, nbAxon=nbHyp
                    self.layers.append(Layer(i+1, nbNeuronsPerLayer, nbHypotheses, nbInputs, activationFunction))
                else:
                    self.layers.append(Layer(i+1, nbNeuronsPerLayer, nbNeuronsPerLayer, nbNeuronsPerLayer, activationFunction))
            else:
                if i == nbHiddenLayers - 1: #si c'est la dernière couche cachée..
                    self.layers.append(Layer(i+1, nbNeuronsPerLayer, nbHypotheses, nbNeuronsPerLayer, activationFunction))
        #Hypotheses layer
        if nbHiddenLayers <= 0:
            self.layers.append(Layer(1, nbHypotheses, 1, nbInputs, activationFunction))
        else:
            self.layers.append(Layer(self.nbHiddenLayers+1, nbHypotheses, 1, nbNeuronsPerLayer, activationFunction))

    def initW_b(self):
        for l,layer in enumerate(self.layers):
            if l == 0: #for entry layer...
                layer.initEntryLayer() #..weights are '1'
            else:
                layer.initRandomWeightsAndBias()

    def forwardPropagation(self, inX):
        currEntry = inX
        for l, layer in enumerate(self.layers):
            if l == 0: #entry propagation
                output = layer.entryPropagation(currEntry)
                print("Output of entry=", output)
            else:
                #print("Here, for (", l, ") entry=", currEntry)
                output = layer.layerPropagation(currEntry) #Signal propagation and recup. of output
            currEntry = output
        self.outputState = output #output of the

    def computeErrors(self, outputs, y):
        #TODO : body
        deltas = []
        return deltas


    def learnExample(self, x, y):
        outputs = self.forwardPropagation(x) #Matrix of ALL output values
        deltas = self.computeErrors(outputs, y) #calcul des deltas
        derivatives = self.computeDerivatives(deltas, outputs)
        self.updateW_b(derivatives)

    def computeDerivatives(self, deltas, outputs):
        #TODO : body
        derivatives = []
        return derivatives

    def learning(self, desiredOutput): #For gate example
        self.layers[1].learn(desiredOutput)

    def printPerceptron(self):
        for l, layer in enumerate(self.layers):
            print("Layer #", l)
            layer.printLayer()