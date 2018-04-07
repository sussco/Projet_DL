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

    def __init__(self, nbfilters = 1, entryW=4, entryH=4, entryD=3, filterSize = 3, stride=1, zeroPad=1, lr = 0.2):

        # Dimensions de l'image d'entree
        self.entryH = entryH
        self.entryW = entryW
        self.entryD = entryD

        #Zeropadding : nombre de couches de zeros autour de l'image
        self.zeroPad = zeroPad
        #Pas de la convolution
        self.stride = stride

        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)


        #Stack of ConvLayer2D
        self.conv2Dlayers = []
        for l in range(nbfilters):
            self.conv2Dlayers.append(ConvLayer2D(entryW, entryH, entryD, filterSize, stride, zeroPad))

        self.learningRate = lr


    def propagation(self, previousLayer):

        for layer2d in self.conv2Dlayers:
            layer2d.feedforward(previousLayer)





    def feedback(self, dH):
        def feedback(self, dH):
            """
            Learning step.
            :param dH: tab of derivatives of the next layer (supposed that a convolution is never the last layer block)
            :return: dX, gradient of the cost of
            """
        try:
            dH.shape()
        except AttributeError as mess:
            print("Seems that partial derivative of layer l+1 is not given as a np.array : {0}".format(mess))

        if dH.shape() != (self.nbfilters, 0, 0): #TODO: heeeeeeeeeeere
            pass



    def computeDeltaTable(self, nextLayers, prevLayers):
        for i in range(len(self.conv2Dlayers)):
            self.conv2Dlayers[i].computeDeltaTable(nextLayers[i], prevLayers[i])

    def computeWeightsTable(self, prevLayer, deltaTables):
        for i in range(len(self.conv2Dlayers)):
            self.conv2Dlayers[i].computeWeightsTable(prevLayer, deltaTables[::,::,i])

    def updateParams(self, nbTrainings):
        for layer2D in self.conv2Dlayers:
            layer2D.updateParams(nbTrainings)


    """
        CIFAR-10 format : 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    """



class ConvLayer2D():

    def __init__(self, entryW=4, entryH=4, entryD=3, filterSize = 3, stride=1, zeroPad=1, lr = 0.2): #For 1 channel = for 1 dim im

        # Calcul des dimensions de la sortie
        # depend de zeroPad et de stride
        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)

        # initialisation de la matrice de sortie
        self.activationTable = np.zeros( (self.layH, self.layW, entryD) )

        #Initialisation of weights and bias
        self.filterWeights = np.random.uniform(0, 0.05, size = (filterSize, filterSize, entryD)) #Shared weights
        self.filterBias = np.random.uniform(0, 0.05, size = (self.layH, self.layW, entryD))
        #Initialisation des matrices d'erreurs
        self.filterWeightsTable = np.zeros(shape = (filterSize, filterSize, entryD))
        self.filterBiasTable = np.zeros(shape = (self.layH, self.layW, entryD))
        # initialisation de la table des deltas
        self.deltaTable = np.zeros(shape = (self.layW, self.layH, entryD))
        # Dimensions de l'image d'entree
        self.entryH = entryH
        self.entryW = entryW
        self.entryD = entryD

        #Zeropadding : nombre de couches de zeros autour de l'image
        self.zeroPad = zeroPad
        #Pas de la convolution
        self.stride = stride

        # Image modifiee, c'est a dire avec le zeropadding
        self.modEntry = np.zeros( shape = (self.layW +2*self.zeroPad, self.layH +2*self.zeroPad, entryD) )

    #    print("Weights init values : w=",self.filterWeights)

        self.learningRate = lr


    def feedforward(self, prevLayer):
        # on copie l'image a traiter, elle va etre modifiee
        imageCp = deepcopy(prevLayer)
        # ajout de zeros autour de l'image depend de l'entier zeroPad
        for k in range(self.zeroPad):
            imageCp = np.insert(imageCp, imageCp.shape[0], 0, axis = 0)
            imageCp = np.insert(imageCp, imageCp.shape[1], 0, axis = 1)
            imageCp = np.insert(imageCp,0,0, axis = 1)
            imageCp = np.insert(imageCp,0,0, axis = 0)
        # calcul de la sortie
        self.modEntry = imageCp
        # pour chaque couleur
        for channel in  range(prevLayer.shape[2]):
            # i: lignes
            for i in range(0, imageCp.shape[0]-2*self.zeroPad, self.stride):
                # j : colonnes
                for j in range(0, imageCp.shape[1]-2*self.zeroPad, self.stride):
                    # on calcule la sortie (self.activationTable)
                    self.activationTable[i,j, channel] =sigmoid((np.multiply(
                    # le morceau de l'image a convoluer
                    imageCp[i: i+len(self.filterWeights[0]),
                    j: j+len(self.filterWeights[1]),
                    channel],
                    # le filtre
                        np.rot90(self.filterWeights[::, ::, channel], 4))).sum()) # rot180 pour faire une convolution et pas un correlation
                    # le biais
                    #+ self.filterBias[i,j,channel])



    def computeDeltaTable(self, nextLayer, prevLayer):

        #deltaTable : table des erreurs des activationTables (deltas)
        #weightsTable : table des erreurs des poids

        dH = nextLayer.deltaTable


        if dH.shape() != self.activationTable.shape():
            print("FEEDBACK ERROR : dH has dim {0} instead of layer shape = {1}".format(dH.shape, self.activationTable.shape))
            exit()
        #Pour chaque couleur
        for channel in range(dH.shape[2]):
            for i in range(dH.shape[0]):
                for j in range(dH.shape[1]):
                    #Calcul de self.deltaTable
                    self.deltaTable[i,j,channel] =  np.multiply(
                    # la deltaTable de la couche suivante
                    dH[i: i+len(self.filterWeights[0]),
                    j: j+len(self.filterWeights[0]),
                    channel],
                    # Les poids de la couche suivante
                    np.rot90(nextLayer.filterWeights[channel],4)
                    # derivee de la sigmoide
                    ) * prevLayer[i,j,channel]*(1-prevLayer[i,j,channel])





    def computeWeightsTable(self, prevLayer, deltaTable):
        #print(prevLayer.shape)
        deltaTable = np.reshape(deltaTable, (28,28,1))
        if self.activationTable.shape != deltaTable.shape:
            print("bad format")
            exit()

        """for channel in range(prevLayer.shape[2]):
            for m in range(self.filterWeightsTable.shape[0]):
                for n in range(self.filterWeightsTable.shape[1]):
                    for i in range(deltaTable.shape[0]):
                        for j in range(deltaTable.shape[1]):
                            self.filterWeightsTable[m,n,channel] += self.deltaTable[i,j,channel]*prevLayer[i+m, j+n, channel]
                            """
        # ca c'est mieux, verifier les indices
        for channel in range(prevLayer.shape[2]):
            for m in range(self.filterWeightsTable.shape[0]):
                for n in range(self.filterWeightsTable.shape[1]):
                    self.filterWeightsTable[m,n,channel] += np.multiply(
                    deltaTable[::, ::, channel], self.modEntry[m: m+self.layW, n: n+self.layH, channel]
                    ).sum()








    def updateParams(self, nbTrainings):
        for channel in range(self.filterWeightsTable.shape[2]):
                    self.filterWeights[::, ::, channel] -= self.learningRate * ( 1/float(nbTrainings) * self.filterWeightsTable[::, ::, channel])
                    self.filterWeightsTable[::, ::, channel] = 0
                    self.filterBias[::, ::, channel] -= self.learningRate * ( 1/float(nbTrainings) * self.filterBiasTable[::, ::, channel])
                    self.filterBiasTable[::, ::, channel] = 0

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
