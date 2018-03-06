# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

tableau = np.loadtxt("/home/mamene/Documents/cours/Projet_file/python-mnist/resulttrie2.res") #on ouvre le fichier avec les résultats

plt.plot(tableau)
plt.show()
