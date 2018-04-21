import numpy as np
from random import uniform
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    try:
        ans = 1 / (1 + math.exp(-x))
    except OverflowError:
        ans = float('inf')
    return ans

def reLU(x):
    if(x>0):
        return x
    else:
        return 0.01*x

def backProp_ReLU(x):
    if x>0:
        return 1
    else:
        return 0.01


class ConvLayer():

    def __init__(self, nbFilters = 1, entryW=4, entryH=4, entryD=3,filterSize = 3, filterD = 1, stride=1, zeroPad=1, lr = 0.2): #For 1 channel = for 1 dim im

        # Calcul des dimensions de la sortie
        # depend de zeroPad et de stride
        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)
        self.inShape = (entryD, entryH, entryW)
        self.zeroPad = zeroPad      #Zeropadding : nombre de couches de zeros autour de l'image
        self.stride = stride        #Pas de la convolution
        self.learningRate = lr
        self.nbFilters = nbFilters
        self.filterSize = filterSize


        self.filterErrors = np.zeros(size = (nbFilters, entryD, filterSize, filterSize))
        self.filterTable =np.random.uniform(0, 0.05, size = (nbFilters, entryD, filterSize, filterSize)))
        self.bias = np.random.uniform(0, 0.05, shape = (nfFilters))



            #self.biasErrors.append( np.zeros(shape = (entryD, self.layH, self.layW)))





    def propagation(self, prevLayer):
        # on copie l'image a traiter, elle va etre modifiee
        #prevLayer = np.reshape(prevLayer, (self.nbFilters, ))
        inPut = deepcopy(prevLayer)
        self.activationTable = np.zeros( (nbFilters, self.layH, self.layW) )
        # ajout de zeros autour de l'image depend de l'entier zeroPad
        for k in range(self.zeroPad):
            inPut = np.insert(inPut, inPut.shape[1], 0, axis = 1)
            inPut = np.insert(inPut, inPut.shape[2], 0, axis = 2)
            inPut = np.insert(inPut,0,0, axis = 1)
            inPut = np.insert(inPut,0,0, axis = 2)

        self.inPut = inPut
        for filters in range(self.nbFilters):
            for i in range(0, self.layH , self.stride):
                for j in range(0, self.layW, self.stride):
                    self.activationTable[filters, i, j] =
                    np.sum(inPut[:,i: i+self.filterSize,
                    j: j+self.filterSize] *
                    self.filterTable[filters]) # rot180 pour faire une convolution et pas un correlation
                    + self.bias[filters])
        return self.activationTable

    def computeDeltaTable(self, nextDeltaTable):
        nextDeltaTable = np.reshape(nextDeltaTable, (self.nbFilters, , self.layW,self.layH))
        self.deltaTable = np.zeros(shape = (entryD, self.entryH, self.entryW)))
        for channel in range(self.inShape[0]):
            for filters in range(self.nbFilters):
            #Pour chaque couleur
                for i in range(self.layH):
                    for j in range(self.layW):
                        #Calcul de self.deltaTable
                        self.deltaTable[channel, i: i+self.filterSize,j: j+self.filterSize]
                        += nextDeltaTable[filters, channel, i, j] * self.filterTable[filters][channel]


    def computeWeightsTable(self, nextDeltaTable):
            for filters in range(self.nbFilters):
                for channel in range(self.inShape[0]):
                    for m in range(self.layH):
                        for n in range(self.layW):
                            self.filterErrors[filters,channel] +=
                            NextDeltaTable[filters,channel, m, n] *
                            self.inPut[channel, m: m+self.filterSize, n: n+self.filterSize]

    def computeBiasTable(self, nextDeltaTable):
        self.biasErrors = np.zeros(shape = (nfFilters))
        for filters in range(nbFilters):
            self.biasErrors[filters] = np.sum(nextDeltaTable[filter])

    def backPropagation(self, nextDeltaTable):
        self.computeWeightTable(nextDeltaTable)
        self.computeBiasTable(nextDeltaTable)
        return self.computeDeltaTable(nextDeltaTable)


    def updateParams(self, nbTrainings):
        for filters in range(self.nbFilters):
            for channel in range(self.entryD):
                        #print(self.filterTable[filters][channel])
                        self.filterTable[filters][channel] -= self.learningRate * ( 1/float(nbTrainings) * self.filterErrors[filters][channel])
                        self.filterErrors[filters][ channel] = 0
                        self.bias[filters] -= self.learningRate * ( 1/float(nbTrainings) * self.biasErrors[filters])
                        self.biasErrors[filters] = 0
            #print(self.filterTable[filters])
