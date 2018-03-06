import numpy as np

class Neuron:

    def __init__(self, layer, index, nbDendrites, activationFunction):
        self.layer = layer
        self.index = index
        self.nbDendrites = nbDendrites
        self.activationFunction = activationFunction #String

        self.entries = []
        self.state = 0
        self.weights = []
        self.bias = 0
        self.z = 0
        self.delta = 0

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def setW_b(self, weights, bias):
        if len(weights) != self.nbDendrites:
            print("Error initialising neuron #", self.index, " : len(weights)=", len(self.weights), "and", "self.nbD=", self.nbDendrites)
            exit();
        self.weights = weights
        self.bias = bias

    def computeState(self, entries):
        for i, xi in enumerate(entries):
            self.z += self.weights[i]*xi
        self.z += self.bias
        if self.activationFunction == "sigmoid":
            self.state = self.sigmoid(self.z)
        else:
            #TODO: implement other activation functions
            pass
        return self.state

    def computeOutputDelta(self, desiredOutputi):
        if self.activationFunction == "sigmoid":
            zi = self.z
            fprimeOf_zi_l = self.sigmoid(zi)(1-self.sigmoid(zi)) #property of sigmoid
            deltai_nl = -(desiredOutputi-self.state)*fprimeOf_zi_l
            self.delta = deltai_nl
            return self.delta
        return self.delta

    def computeHiddenDelta(self, tabDeltas_lplus1):
        """Calculates the error of the neuron (for retropropagation)"""
        if self.activationFunction == "sigmoid":
            zi = self.z
            fprimeOf_zi_l = self.sigmoid(zi)(1-self.sigmoid(zi)) #property of sigmoid
            sum = 0
            for j, deltaj_lplus1 in enumerate(tabDeltas_lplus1):
                wl_ij = self.weights[j]
                sum += wl_ij*deltaj_lplus1
            self.delta = sum * fprimeOf_zi_l
        else:
            #TODO : other activation functions
            pass
        return self.delta

    def tanh(self, entries):
        pass
    def max(self, entries):
        #TODO : implement
        pass


    def activate(self, entries):
        self.entries = entries
        if self.activationFunction == "trigger":
            #print("Activation neuron") #", self.layer, self.index, "with w=", self.weights, "b=", self.bias, "and inputs=", entries)
            z = np.dot(self.weights, entries) - self.bias
            if z>0:
                self.state = 1
            else:
                self.state = 0
        elif self.activationFunction == "entry":
            self.state = entries
        return self.state

    def updateW_b_old(self, desiredOutput):
        for i, w in enumerate(self.weights):
            #print("1 w=",w, self.weights[i])
            w = w + 0.1*(desiredOutput[0]-self.state)*self.entries[i][0]
            #print("2 w=",w, self.weights[i])
            self.weights[i] = w
            #print("3 weights=", self.weights)
        print("Old bias=", self.bias)
        self.bias = self.bias + 0.3*(desiredOutput[0]-self.state)
        print("New bias =",self.bias)

    def updateW_b(self, learningRate, partialDerivativeOfJ_onWijl, partialDerivativeOfJ_onbil):
        for i, w in enumerate(self.weights):
            w -= learningRate * partialDerivativeOfJ_onWijl
        self.bias -= learningRate * partialDerivativeOfJ_onbil

    def printNeuron(self):
        print("   Neuron #", self.layer, self.index, "weights=", self.weights, "bias=", self.bias, "state=", self.state)