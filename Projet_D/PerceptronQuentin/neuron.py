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

    def setW_b(self, weights, bias):
        if len(weights) != self.nbDendrites:
            print("Error initialising neuron #", self.index, " : len(weights)=", len(self.weights), "and", "self.nbD=", self.nbDendrites)
            exit();
        self.weights = weights
        self.bias = bias

    def stigmoid(self, entries):
        pass
    def tanh(self, entries):
        pass
    def max(self, entries):
        #TODO : implement
        pass

    def learn(self, desiredOutput):
        for i, w in enumerate(self.weights):
            #print("1 w=",w, self.weights[i])
            w = w + 0.1*(desiredOutput[0]-self.state)*self.entries[i][0]
            #print("2 w=",w, self.weights[i])
            self.weights[i] = w
            #print("3 weights=", self.weights)
        print("Old bias=", self.bias)
        self.bias = self.bias + 0.3*(desiredOutput[0]-self.state)
        print("New bias =",self.bias)

    def activate(self, entries):
        self.entries = entries
        if self.activationFunction == "trigger":
            #print("Activation neuron #", self.layer, self.index, "with w=", self.weights, "b=", self.bias, "and inputs=", entries)
            z = np.dot(self.weights, entries) - self.bias
            if z>0:
                self.state = 1
            else:
                self.state = 0
        elif self.activationFunction == "entry":
            self.state = entries
        return self.state

    def printNeuron(self):
        print("   Neuron #", self.layer, self.index, "weights=", self.weights, "bias=", self.bias, "state=", self.state)