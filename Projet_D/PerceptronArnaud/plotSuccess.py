import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("list1LR","rb") as f:
    x1 = pickle.load(f)
with open("list1SR","rb") as f:
    y1 = pickle.load(f)

with open("list10LR","rb") as f:
    x10 = pickle.load(f)
with open("list10SR","rb") as f:
    y10 = pickle.load(f)

with open("list100LR","rb") as f:
    x100 = pickle.load(f)
with open("list100SR","rb") as f:
    y100 = pickle.load(f)


fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.scatter(x1,y1, marker = "s", label = "batch = 1")
plt.scatter(x10,y10, marker = "o", label = "batch = 10")
plt.scatter(x100,y100, marker = "*", label = "batch = 100")

plt.show()
