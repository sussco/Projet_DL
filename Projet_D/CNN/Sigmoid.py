
import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def propagation(self, input):

        self.inPut = input
        self.activationTable = 1.0/(1.0 + np.exp(-input))
        return self.activationTable

    def backPropagation(self, nextDeltaTable):

        self.deltaTable = nextDeltaTable * (self.activationTable) * (1 - self.activationTable)
        return self.deltaTable
