from layer import *
import numpy as np
from random import uniform

class ConvLayer(Layer):

    def __init__(self, nbfilters, inputSizeXYZ, zeroPaddingXY=1, strideXY=1, regionSizeXY=[3,3]):
        """

        :param stride: stride for convolution, typically 1 or 2
        :param nbfilters:
        :param regionSizeXY: region to be filtered by ONE neuron
        :param inputSizeXYZ: [w, h, d] of the input layer (or image)
        :param zeroPaddingXY: [paddingX, paddingY]
        """
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

    def __init__(self, layW, layH, entryH=4, entryW=4, entryD=3, stride=1): #For 1 channel = for 1 dim im
        #Change default values for entry
        self.filterWeights = np.zeros((entryD, 3*3)) #Shared weights. TODO : calcultate layH & layW = fct(F,S,P..)
        self.layerState = np.zeros((layH, layW)) #TODO: calcultate layH & layW = fct(F,S,P..)
        self.patches = []

        self.entryH = entryH
        self.entryW = entryW
        self.entryD = entryD

        self.layW = layW
        self.layH = layH

        #init weights & bias randomly
        #                     |Row            |Line
        # filterWaights = [ [w00, w01, .. , w0x, w10, .. , wyx], .....   [w00, w01, .. , w0n, w10, .. , wyx]  ]
        for channelWeightsTab in self.filterWeights:
            for i in range(len(np.array(channelWeightsTab).flatten())):
                channelWeightsTab[i] = uniform(0, 0.01) #TODO: Good value ? -> tests
        print("Weights init values : w=",self.filterWeights)
        self.filterBias = uniform(0, 0.01)

    def feedforward(self, prevLayer, prevLayW, prevLayH, prevLayD=3):
        """

        :param prevLayer: format  np.array[ [1, 1 ,1],
                                            [1, 1, 1],
                                            [1, 1, 1]

                                            [1, 1 ,1],
                                            [1, 1, 1],
                                            [1, 1, 1] ], ..)
        :return: self.layer = np.array( [ [ ., ., . ],
                                          [ ., ., . ],
                                          [ ., ., . ] ] )
        """
        print("Previous layer to compute : ",prevLayer)

        #Creating patches for convolution
        #self.patches = np.array( [ [ array([ [0.  , 0.  , 0.  ],
        #                                     [0.  , 0.11, 0.12],
        #                                     [0.  , 0.21, 0.22] ]),
        #
        #                             array([ [0.  , 0.  , 0.  ],
        #                                     [0.11, 0.12, 0.13],
        #                                     [0.21, 0.22, 0.23]])  ], .... all for ch.1
        #                             array([ [ ., ., . ],
        #                                     [ ., ., . ], ..... ch2 ...
        self.patches = []
        for layerSlide in prevLayer: #for each channel..
            self.patches.append(self.layer2d2col(layerSlide, prevLayW, prevLayH)) #make patches

        #print("Patches :", self.patches)

        channelPatches = []
        nbChannels = len(self.patches)
        for c in range(nbChannels): #separate patches by slides of previous layer
            channelPatches.append(self.patches[c])
            #print("Channel c=", c, patches[c])

        print("Channel patches:", channelPatches)
        print("channelPatches[c][n]=", channelPatches[0][4])

        #Computing layer state
        for l, line in enumerate(self.layerState):
            for n, neuron in enumerate(line):
                neuron = 0
                for c in range(nbChannels):
                    print("\nPatch token on channel={0} on {1} : {2}".format(c, nbChannels, np.array(channelPatches[c][4*l+n]).flatten()))
                    print("for this channel, weights=", np.array(self.filterWeights[c]), len(np.array(self.filterWeights[c])))

                    neuron += np.dot(np.array(channelPatches[c][4*l+n]).flatten(), np.array(self.filterWeights[c]).flatten()) #TODO: adjust indice '4*' with param size
                    print("neuron=", neuron)
                neuron += self.filterBias
                print("Before value neuron=", neuron)
                neuron = ConvLayer.activationFunction(neuron)
                print("Final value neuron=", neuron)
                self.layerState[l][n] = neuron

            #print(self.patches)
            #for patch in self.patches:
            #    print(patch)
        print("Layer state:", self.layerState)
        return self.layerState

    def feedback(self, dH):
        """
        Learning step.
        :param dH: tab of derivatives of the next layer (supposed that a convolution is never the last layer block)
        :return: dX, gradient of the cost of
        """
        try:
            dH.shape()
        except AttributeError as mess:
            print("FEEDBACK ERROR : Seems that partial derivative of layer l+1 is not given as a np.array : {0}".format(mess))

        if dH.shape() != self.layerState.shape():
            print("FEEDBACK ERROR : dH has dim {0} instead of layer shape = {1}".format(dH.shape, self.layerState.shape))
            exit()

        #Computing derivatives : for each neuron, dWij = scalar(patchX, dH)
        for ij in len(self.layerState.flatten()):
            dWij = 0
            for c in self.entryD: #Don't forget to sum on all channels
                dWij += np.dot(dH.flatten(), np.array(self.patches[c][ij]).flatten())
                #TODO: check
            print("dWij = {0}".format(dWij))


    def layer2d2col(self, layer2d, layW, layH, regionSize=3): #TODO:extend zeropadding, stride
        """
        Works for 3x3 patches
        :param layer2d: must be a np.array((xsize, ysize)) in 1 dimension (1 channel)
        :param layWidth:
        :param height:
        :param regionSize: size of the patch thats convolves
        :return: patches (np.array((width, size))
        """
        transforming = True
        patches = []
        x,y = 0, 0
        while transforming :
            currPatch = np.zeros((regionSize, regionSize))
            #print("x={0} y={1} layW={2} layH={3}".format(x, y, layW, layH))

            if y == 0 and x == 0: ##First row
                # 0 0 0
                # 0 . .
                # 0 . .
                #Zero-padding of first line, ok
                #zero-padding of (1,0), ok
                #Catching "pixels"
                currPatch[1][1] = layer2d[0][0]
                currPatch[1][2] = layer2d[0][1]

                currPatch[2][1] = layer2d[1][0]
                currPatch[2][2] = layer2d[1][1]
                x+=1
            elif y==0 and (0 < x and x < layW-1):
                # 0 0 0
                # . . .
                # . . .
                currPatch[1][0] = layer2d[y][x-1]
                currPatch[1][1] = layer2d[y][x]
                currPatch[1][2] = layer2d[y][x+1]
                currPatch[2][0] = layer2d[y+1][x-1]
                currPatch[2][1] = layer2d[y+1][x]
                currPatch[2][2] = layer2d[y+1][x+1]
                x+=1
            elif y == 0 and x == layW-1:
                # 0 0 0
                # . . 0
                # . . 0
                #Zero-padding of first line, ok
                #zero-padding of (1, layW-1), ok
                #Catching "pixels"
                currPatch[1][0] = layer2d[y][x-1]
                currPatch[1][1] = layer2d[y][x]

                currPatch[2][0] = layer2d[y+1][x-1]
                currPatch[2][1] = layer2d[y+1][x]
                y+=1
                x = 0
            elif (0 < y and y < layH-1) and x == 0:
                # 0 .  .
                # 0 0y .
                # 0 .  .
                currPatch[0][1] = layer2d[y-1][x]
                currPatch[0][2] = layer2d[y-1][x+1]
                currPatch[1][1] = layer2d[y][x]
                currPatch[1][2] = layer2d[y][x+1]
                currPatch[2][1] = layer2d[y+1][x]
                currPatch[2][2] = layer2d[y+1][x+1]
                x+=1
            elif (0 < y and y < layH-1) and x == layW-1:
                # . .  0
                # . 0y 0
                # . .  0
                currPatch[0][0] = layer2d[y-1][layW-2]
                currPatch[0][1] = layer2d[y-1][layW-1]
                currPatch[1][0] = layer2d[y][layW-2]
                currPatch[1][1] = layer2d[y][layW-1]
                currPatch[2][0] = layer2d[y+1][layW-2]
                currPatch[2][1] = layer2d[y+1][layW-1]
                y+=1
                x = 0
            elif y == layH-1 and x == 0:
                # 0 . .
                # 0 . .
                # 0 0 0
                currPatch[0][1] = layer2d[y-1][0]
                currPatch[0][2] = layer2d[y-1][1]

                currPatch[1][1] = layer2d[y][0]
                currPatch[1][2] = layer2d[y][1]
                x+=1
            elif y==layH-1 and (0 < x and x < layW-1):
                # . . .
                # . . .
                # 0 0 0
                currPatch[0][0] = layer2d[y-1][x-1]
                currPatch[0][1] = layer2d[y-1][x]
                currPatch[0][2] = layer2d[y-1][x+1]
                currPatch[1][0] = layer2d[y][x-1]
                currPatch[1][1] = layer2d[y][x]
                currPatch[1][2] = layer2d[y][x+1]
                x+=1
            elif y == layH-1 and x == layW-1:
                # . . 0
                # . . 0
                # 0 0 0
                currPatch[0][0] = layer2d[y-1][x-1]
                currPatch[0][1] = layer2d[y-1][x]

                currPatch[1][0] = layer2d[y][x-1]
                currPatch[1][1] = layer2d[y][x]
                transforming = False #finished convoluting
            elif 0 < y and y < layH-1 and 0 < x and x < layW-1:
                # . .  .
                # . xy .
                # . .  .
                currPatch[0][0] = layer2d[y-1][x-1]
                currPatch[0][1] = layer2d[y-1][x]
                currPatch[0][2] = layer2d[y-1][x+1]
                currPatch[1][0] = layer2d[y][x-1]
                currPatch[1][1] = layer2d[y][x]
                currPatch[1][2] = layer2d[y][x+1]
                currPatch[2][0] = layer2d[y+1][x-1]
                currPatch[2][1] = layer2d[y+1][x]
                currPatch[2][2] = layer2d[y+1][x+1]
                x+=1
            else:
                print("Problem in convolution : x={0} y={1}".format(x, y))
                print(patches)
                exit()
            patches.append(currPatch)
        return patches