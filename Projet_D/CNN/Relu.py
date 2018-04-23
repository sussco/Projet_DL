
import numpy as np

class ReLU:
    def __init__(self):
        pass

    def propagation(self, X):
        self.inPut = X
        self.activationTable = np.maximum(X, 0)
        return self.activationTable

    def backPropagation(self, nextDeltaTable):

        self.deltaTable = nextDeltaTable * (self.inPut >= 0)
        return self.deltaTable
