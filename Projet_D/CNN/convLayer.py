import numpy as np
from random import uniform
import matplotlib.image as img
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import scipy.signal

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

    def __init__(self,  entryW=4, entryH=4, entryD=3, nbFilters = 1, filterSize = 3, stride=1, zeroPad=1): #For 1 channel = for 1 dim im

        # Calcul des dimensions de la sortie
        # depend de zeroPad et de stride
        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)
        self.inShape = (entryD, entryH, entryW)
        self.zeroPad = zeroPad      #Zeropadding : nombre de couches de zeros autour de l'image
        self.stride = stride        #Pas de la convolution
        self.nbFilters = nbFilters
        self.filterSize = filterSize


        self.filterErrors = np.zeros(shape = (nbFilters, entryD, filterSize, filterSize))
        self.filterTable =np.random.normal(0, 0.1, size = (nbFilters, entryD, filterSize, filterSize))
        self.bias = np.random.normal(0, 0.1, size = (nbFilters))
        self.biasErrors = np.zeros(shape = (self.nbFilters))

        self.vFilter = np.zeros(shape = (nbFilters, entryD, filterSize, filterSize))
        self.vBias = np.zeros(shape = (self.nbFilters))



    def propagation(self, prevLayer):
        # on copie l'image a traiter, elle va etre modifiee
        #prevLayer = np.reshape(prevLayer, (self.nbFilters, ))
        self.activationTable = np.zeros( (self.nbFilters, self.layH, self.layW) )
        self.inPut =  prevLayer

        padded_input = np.pad(prevLayer, ((0,0), (self.zeroPad, self.zeroPad), (self.zeroPad, self.zeroPad)), 'constant')
        rotated_filter = np.rot90(self.filterTable, 2, (2,3))

        for filters in range(self.nbFilters):
            for input_depth in range(self.inShape[0]):
                self.activationTable[filters] += scipy.signal.convolve2d(padded_input[input_depth], rotated_filter[filters, input_depth], mode='valid')
            self.activationTable[filters] += self.bias[filters]
        # print(self.activationTable.shape)
        # print("HA\n", self.activationTable[0][6])
        return self.activationTable

    def computeDeltaTable(self, nextDeltaTable):
        # on pad l'entrée à nouveau si elle l'a été lors de la propagation
        padded_input = np.pad(self.inPut, ((0,0), (self.zeroPad, self.zeroPad), (self.zeroPad, self.zeroPad)), 'constant')
        self.deltaTable = 0
        # la table des deltas avec le Zeropadding
        deltaPadded = np.zeros(shape = padded_input.shape)
        for filters in range(self.nbFilters): # iteration sur les filtres
            for i in range(self.layH):
                for j in range(self.layW):
                    # calcul du deltaTable
                    deltaPadded[:, i: i+self.filterSize,j: j+self.filterSize]+= nextDeltaTable[filters, i, j] * self.filterTable[filters]
        if(self.zeroPad != 0):
            self.deltaTable = deltaPadded[:, self.zeroPad:-self.zeroPad, self.zeroPad:-self.zeroPad]
        else :
            self.deltaTable = deltaPadded
        return self.deltaTable

    def computeWeightsTable(self, nextDeltaTable):
        padded_input = np.pad(self.inPut, ((0,0), (self.zeroPad, self.zeroPad), (self.zeroPad, self.zeroPad)), 'constant')
        for filters in range(self.nbFilters):
            # for input_depth in range(self.inShape[0]):
                for m in range(self.layW):
                    for n in range(self.layH):
                        self.filterErrors[filters,:,:,:] += nextDeltaTable[filters, m, n]*padded_input[:, m: m+self.filterSize, n: n+self.filterSize]

                        # self.filterErrors[filters, input_depth, m, n] += np.multiply(
                        # nextDeltaTable[filters],
                        # padded_input[input_depth, m: m+self.layW, n: n+self.layH]
                        # ).sum()


    def computeBiasTable(self, nextDeltaTable):
        for filters in range(self.nbFilters):
            self.biasErrors[filters] = np.sum(nextDeltaTable[filters, :, :])

    def backPropagation(self, nextDeltaTable):
        self.computeWeightsTable(nextDeltaTable)
        self.computeBiasTable(nextDeltaTable)
        return self.computeDeltaTable(nextDeltaTable)


    def updateParams(self, nbTrainings, learningR):
        for filters in range(self.nbFilters):
            self.filterTable[filters]-= learningR * ( 1/float(nbTrainings) * self.filterErrors[filters])
            self.filterErrors[filters] = 0
            self.bias[filters] -= learningR * ( 1/float(nbTrainings) * self.biasErrors[filters])
            self.biasErrors[filters] = 0
        # print(self.filterTable[0][0])


    def updateParamsMomentum(self, nbTrainings, learningR, momentum):
        for filters in range(self.nbFilters):
            self.vFilter[filters] = momentum*self.vFilter[filters]+ learningR* ( 1/float(nbTrainings)) *self.filterErrors[filters]
            self.vBias[filters] = momentum*self.vBias[filters]+ learningR* ( 1/float(nbTrainings)) *self.biasErrors[filters]

            self.filterTable[filters]-= self.vFilter[filters]
            self.bias[filters] -= self.vBias[filters]

            self.filterErrors[filters] = 0
            self.biasErrors[filters] = 0
        # print(self.filterTable[0][0])
