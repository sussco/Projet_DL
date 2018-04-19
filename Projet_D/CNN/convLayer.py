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




class ConvLayer():

    def __init__(self, nbFilters = 1, entryW=4, entryH=4, entryD=3,filterSize = 3, stride=1, zeroPad=1, lr = 0.2): #For 1 channel = for 1 dim im

        # Calcul des dimensions de la sortie
        # depend de zeroPad et de stride
        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)
        self.entryH = entryH
        self.entryW = entryW
        self.entryD = entryD
        self.zeroPad = zeroPad      #Zeropadding : nombre de couches de zeros autour de l'image
        self.stride = stride        #Pas de la convolution
        self.learningRate = lr
        self.nbFilters = nbFilters
        self.filterSize = filterSize

        self.activationTable = []
        self.filterTable = []
        self.biasTable = np.random.uniform(0, 0.05)
        self.filterErrors = []
        self.biasErrors = 0
        self.deltaTable = []


        for i in range(nbFilters):
            self.filterTable.append(np.random.uniform(0, 0.05, size = (entryD, filterSize, filterSize)))
            #self.biasTable.append(np.random.uniform(-0.05, 0.05, size = (entryD, self.layH, self.layW)))
            self.filterErrors.append( np.zeros(shape = (entryD, filterSize, filterSize)))
            #self.biasErrors.append( np.zeros(shape = (entryD, self.layH, self.layW)))
            self.deltaTable.append( np.zeros(shape = (entryD, self.entryH, self.entryW)))
            self.activationTable.append( np.zeros( (entryD, self.layH, self.layW) ))
        self.modEntry = np.zeros( shape = (entryD, self.layW +2*self.zeroPad, self.layH +2*self.zeroPad) )





    def propagation(self, prevLayer):
        # on copie l'image a traiter, elle va etre modifiee
        #prevLayer = np.reshape(prevLayer, (self.nbFilters, ))
        imageCp = deepcopy(prevLayer)
        # ajout de zeros autour de l'image depend de l'entier zeroPad
        for k in range(self.zeroPad):
            imageCp = np.insert(imageCp, imageCp.shape[1], 0, axis = 1)
            imageCp = np.insert(imageCp, imageCp.shape[2], 0, axis = 2)
            imageCp = np.insert(imageCp,0,0, axis = 1)
            imageCp = np.insert(imageCp,0,0, axis = 2)

        # calcul de la sortie
        # TODO: changer la shape dans la liste des images
        #imageCp = np.reshape(imageCp, (self.entryD, self.layW+2*self.zeroPad, self.layH+2*self.zeroPad))
        self.modEntry = imageCp
        # pour chaque couleur
        for filters in range(self.nbFilters):
            for filterPrev in range(prevLayer.shape[0]):
                for channel in  range(prevLayer.shape[1]):
                    # i: lignes
                    for i in range(0, imageCp.shape[2]-2*self.zeroPad-2, self.stride):
                        # j : colonnes
                        for j in range(0, imageCp.shape[3]-2*self.zeroPad-2, self.stride):
                            # on calcule la sortie (self.activationTable)
                            self.activationTable[filters+filterPrev][channel, i,j] =sigmoid((np.multiply(
                            # le morceau de l'image a convoluer
                            imageCp[filterPrev, channel,i: i+self.filterSize,
                            j: j+self.filterSize],
                            # le filtre
                            np.rot90(self.filterTable[filters][channel], 0)).sum())) # rot180 pour faire une convolution et pas un correlation
                            # le biais
                            #+ self.biasTable)



    def computeDeltaTable(self, nextDeltaTable, prevlayer):
        nextDeltaTable = np.reshape(nextDeltaTable, (self.nbFilters, 1, self.layW,self.layH))
        #copie de la table, on va la modifier
        deltaTableMod = deepcopy(nextDeltaTable)

        #on met des zeros autour en fonction de la taille des filtres
        for k in range(self.filterSize-1):
            deltaTableMod = np.insert(deltaTableMod, deltaTableMod.shape[2], 0, axis = 2)
            deltaTableMod = np.insert(deltaTableMod, deltaTableMod.shape[3], 0, axis = 3)
            deltaTableMod = np.insert(deltaTableMod,0,0, axis = 2)
            deltaTableMod = np.insert(deltaTableMod,0,0, axis = 3)
        # print("deltatable L : ", np.array(self.deltaTable).shape)
        # print("deltatable L+1 : ", np.array(deltaTableMod).shape)

        #deltaTable : table des erreurs des activationTables (deltas)
        # prevLayer : les activations de la couche l
        for filters in range(self.nbFilters):
            #Pour chaque couleur
            for channel in range(self.entryD):
                for i in range(self.entryH):
                    for j in range(self.entryW):
                        #Calcul de self.deltaTable
                        self.deltaTable[filters][channel, i,j] =  np.multiply(
                        # la deltaTable de la couche suivante
                        deltaTableMod[filters,channel,i: i+self.filterSize,j: j+self.filterSize],
                        np.rot90(self.filterTable[filters][channel], 0)
                        # derivee de la sigmoide
                        ).sum() * prevlayer[filters][channel, i, j]*(1-prevlayer[filters][channel, i, j])
                        #print(self.deltaTable[filters][channel, i,j])
            #print(self.deltaTable[filters])





    def computeWeightsTable(self, deltaTable):

            for filters in range(self.nbFilters):
                # print("mod: ", self.modEntry.shape)
                for channel in range(np.array(self.modEntry).shape[1]):
                    for m in range(self.filterErrors[filters].shape[1]):
                        for n in range(self.filterErrors[filters].shape[2]):
                            self.filterErrors[filters][channel, m, n] += np.multiply(
                            np.rot90(deltaTable[filters,channel],0),
                            self.modEntry[0, channel, m: m+self.layW, n: n+self.layH]
                            ).sum()
                            # print(self.filterErrors[filters][channel, m, n] )
                # self.biasErrors += deltaTable[filters].sum()

    def updateParams(self, nbTrainings):
        for filters in range(self.nbFilters):
            for channel in range(self.entryD):
                        #print(self.filterTable[filters][channel])
                        self.filterTable[filters][channel] -= self.learningRate * ( 1/float(nbTrainings) * self.filterErrors[filters][channel])
                        self.filterErrors[filters][ channel] = 0
                        # self.biasTable -= self.learningRate * ( 1/float(nbTrainings) * self.biasErrors)
                        # self.biasErrors = 0
            #print(self.filterTable[filters])

"""
# Petit test
a = ConvLayer2D(entryW = 1200, entryH = 800)
# Matrice de detection de contour par exemple
a.filterWeights = [[[-1,0,1],[-2,0,2],[-1,0,1]],
[[-1,0,1],[-2,0,2],[-1,0,1]],
[[-1,0,1],[-2,0,2],[-1,0,1]]]
image = img.imread('castle.jpg')
a.feedforward(image)
fig = plt.figure(figsize=(image.shape[0],image.shape[1]))


fig.add_subplot(2,2,1)
plt.imshow(image)
fig.add_subplot(2,2,2)
plt.imshow(a.activationTable[::,::,0])
fig.add_subplot(2,2,3)
plt.imshow(a.activationTable[::,::,1])
fig.add_subplot(2,2,4)
plt.imshow(a.activationTable[::,::,2])
plt.show()
"""
