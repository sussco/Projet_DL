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

class ConvLayer(Layer):

    def __init__(self, nbfilters, inputSizeXYZ, zeroPaddingXY=1, strideXY=1, regionSizeXY=[3,3]):
        """

        :param stride: stride for convolution, typically 1 or 2
        :param nbfilters:
        :param regionSizeXY: region to be filtered by ONE neuron
        :param inputSizeXYZ: [w, h, d] of the input layer (or image)
        :param zeroPaddingXY: [paddingX, paddingY]
        """
        Layer.__init__()

        self.strideXY = strideXY
        self.nbfilters = nbfilters

        self.regionSizeXY = regionSizeXY #Square, rectangle, size of the region filtered by a neuron

        #X = width, Y = height, Z = depth
        self.inputSizeXYZ = inputSizeXYZ
        self.sizeXYZ = np.zeros((inputSizeXYZ[2], inputSizeXYZ[0], inputSizeXYZ[1]))

        inputSizeXY = np.array([inputSizeXYZ[0], inputSizeXYZ[1]]) #size in dim 2
        self.zeroPaddingXY = zeroPaddingXY
        #self.sizeXY = (inputSizeXYZ - regionSizeXY + 2*strideXY)/zeroPaddingXY + 1 #TODO: param for padding..

        #Stack of ConvLayer2D
        self.layD = inputSizeXYZ
        self.convFilters = []
        for l in range(nbfilters):
            self.convFilters.append(ConvLayer2D(inputSizeXYZ[0], inputSizeXYZ[1]))
        self.layerState = [] # tab of tab of tab (3D)


    def propagation(self, previousLayer):
        """For a 2x2x2, outputs a numpy.array( [ [  [ . , . ],
                                                    [ . , . ] ]

                                                  [ [ . , . ]
                                                    [ . , . ] ] ] , ndmin=3"""
        for layer2d in self.convFilters:
            self.layerState.append(layer2d.feedforward(previousLayer, self.inputSizeXYZ[0], self.inputSizeXYZ[1], self.inputSizeXYZ[0]))
        print("\nConv3D state :", self.layerState)
        return self.layerState
        #TODO: heeeeeeeeeeeeeeeeeere: works ?



    def computeError(self):
        pass

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


    def activationFunction(x):
        #TODO: others than sigmoid ?
        return 1/(1+np.exp(-x))


    """
        CIFAR-10 format : 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    """



class ConvLayer2D(Layer):

    def __init__(self, entryW=4, entryH=4, entryD=3, filterSize = 3, stride=1, zeroPad=1): #For 1 channel = for 1 dim im

        # Calcul des dimensions de la sortie
        # depend de zeroPad et de stride
        self.layW = int((entryW-filterSize+2*zeroPad)/stride +1)
        self.layH = int((entryH-filterSize+2*zeroPad)/stride +1)

        # initialisation de la matrice de sortie
        self.layerState = np.zeros( (self.layH, self.layW, entryD) )

        #Initialisation of weights and bias
        self.filterWeights = np.random.uniform(0, 0.05, size = (filterSize, filterSize, entryD)) #Shared weights
        self.filterBias = np.random.uniform(0, 0.05, size = (self.layW, self.layH, entryD))
        #Initialisation des matrices d'erreurs
        self.filterWeightsTable = np.zeros(shape = (filterSize, filterSize, entryD))
        self.filterBiasTable = np.zeros(shape = (self.layW, self.layH, entryD))
        # initialisation de la table des deltas
        self.imageTable = np.zeros(shape = (self.layW, self.layH, entryD))
        # Dimensions de l'image d'entree
        self.entryH = entryH
        self.entryW = entryW
        self.entryD = entryD


        self.zeroPad = zeroPad
        self.stride = stride

        self.modEntry = np.zeros( shape = (self.layW +2*self.zeroPad, self.layH +2*self.zeroPad, entryD) )

        print("Weights init values : w=",self.filterWeights)




    def feedforward(self, prevLayer):
        #assert(prevLayer.shape = ())

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

        for channel in  range(prevLayer.shape[2]):
            for i in range(0, imageCp.shape[0]-2*self.zeroPad, self.stride):
                for j in range(0, imageCp.shape[1]-2*self.zeroPad, self.stride):
                    #print( "i: ", i, "j: ", j)
                    self.layerState[i,j, channel] =(np.multiply(
                    imageCp[i: i+len(self.filterWeights[0]),
                    j: j+len(self.filterWeights[1]),
                    channel],
                    np.rot90(self.filterWeights[channel], 4))).sum() # rot180 pour faire une convolution et pas un correlation
                    #+ self.filterBias[self.layW,self.layH,channel])



    def computeImageTable(self, nextLayer, prevLayer):
        """
        imageTable : table des erreurs des activations
        weightsTable : table des erreurs des poids
        """
        dH = nextLayer.imageTable
        dW = nextLayer.weightsTable

        try:
            dW.shape()
        except AttributeError as mess:
            print("FEEDBACK ERROR : Seems that partial derivative of layer l+1 is not given as a np.array : {0}".format(mess))

        if dH.shape() != self.layerState.shape():
            print("FEEDBACK ERROR : dH has dim {0} instead of layer shape = {1}".format(dH.shape, self.layerState.shape))
            exit()


        #Computing derivatives : for each neuron, dWij = scalar(patchX, dH)
        for channel in range(dH.shape[2]):
            for i in range(dH.shape[0]):
                for j in range(dH.shape[1]):

                    self.imageTable[i,j,channel] =  np.multiply(
                    dH[i: i+len(self.filterWeights[0]),
                    j: j+len(self.filterWeights[0]),
                    channel],
                    nextLayer.filterWeights[channel]
                    ) * sigmoid(prevLayer[i,j,channel])





    def computeWeightsTable(self, prevLayer):

        if prevLayer.shape() != self.imageTable.shape():
            print("FEEDBACK ERROR : dH has dim {0} instead of layer shape = {1}".format(dH.shape, self.layerState.shape))
            exit()

        """for channel in range(prevLayer.shape[2]):
            for m in range(self.filterWeightsTable.shape[0]):
                for n in range(self.filterWeightsTable.shape[1]):
                    for i in range(imageTable.shape[0]):
                        for j in range(imageTable.shape[1]):
                            self.filterWeightsTable[m,n,channel] += self.imageTable[i,j,channel]*prevLayer[i+m, j+n, channel]
                            """
        # ca c'est mieux, à voir si ca marche
        for channel in range(prevLayer.shape[2]):
            for m in range(self.filterWeightsTable.shape[0]):
                for n in range(self.filterWeightsTable.shape[1]):
                    self.filterWeightsTable[m,n,channel] += np.multiply(
                    self.imageTable[channel], self.modEntry[m: m+layW, n: n+layH, channel]
                    )


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
plt.imshow(a.layerState[::,::,0])
fig.add_subplot(2,2,3)
plt.imshow(a.layerState[::,::,1])
fig.add_subplot(2,2,4)
plt.imshow(a.layerState[::,::,2])
plt.show()
