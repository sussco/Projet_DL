from neuron import *
from layer import *
from numpy import dot

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
        self.activationFunction =   activationFunction
        self.nbLayers = nbHiddenLayers + 2
        self.layers = []
        self.outputs = []
        self.deltas = []
        self.costLayerDerivativesW = []
        self.costLayerDerivativesB = []


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

    def computeDeltas(self):
        for l, layer in enumerate(list(reversed(self.layers))): #iterating from end
            if l == len(self.layers)-1: #If it's the output layer, special formula
                self.deltas.append(layer.computeOutputDeltas()) #computation returns a list of each neuron delta
            else:
                self.deltas.append(layer.computeHiddenErrors())
        return self.deltas #list of lists

    def computePartialDerivatives(self):
        #grad/Wl(J(W,b,x,y)) = delta(l+1)*al.T
        for l, layer in enumerate(list(reversed(self.layers.pop(0)))): #for all layers, first expected;
            costDerivativesWl = []
            costDerivativesBl = []
            #for me, Wl is Wl+1 for the ufldl.standford.edu
            for i, deltail in enumerate(layer.deltas):
                costDerivativesWlLinei = []
                costDerivativesBlLinei = []
                for j, alj in enumerate(self.layers[l-1].state): #for each state of neurons l-1
                    costDerivativeWij = alj*deltail
                    costDerivativesWlLinei.append(costDerivativeWij)
                    costDerivativeBli = deltail
                    costDerivativesBlLinei.append(costDerivativeBli)
                costDerivativesWl.append(costDerivativesWlLinei)
                costDerivativesBl.append(costDerivativesBlLinei)
            self.costLayerDerivativesW.append(costDerivativesWl)
            self.costLayerDerivativesB.append(costDerivativesBl)

    def updateW_b(self):
        for l, layer in enumerate(self.layers):
            layer.updateW_b_old(self.costLayerDerivativesW[l])


    def learnExample(self, x, y):
        self.outputs = self.forwardPropagation(x) #Matrix of ALL output values
        self.backpropagation(y)

    #Obsolete
    def computeDerivatives(self, deltas, outputs):
        #TODO : body
        derivatives = []
        return derivatives

    def backpropagation(self, desiredOutput): #For gate example
        self.computeDeltas() #calcul des deltas
        self.computePartialDerivatives()
        self.updateW_b()

    def learnExample(self, input, output):
        self.forwardPropagation(input)
        self.backpropagation(output)

    def printPerceptron(self):
        for l, layer in enumerate(self.layers):
            print("Layer #", l)
            layer.printLayer()