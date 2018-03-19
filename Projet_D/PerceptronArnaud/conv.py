import numpy as np
import inmageTest as im
from pylab import *
import matplotlib.pyplot as plt
import scipy
import matplotlib.image as img
from copy import deepcopy

def conv(image, filterIm, zeroPad, stride):
    # on calcule la taille de li'mage de sortie
    # depend de zeroPad et de stride
    x = (image.shape[0]-len(filterIm[0])+2*zeroPad)/stride +1
    y = (image.shape[1]-len(filterIm[1])+2*zeroPad)/stride +1
    # initialisation de la matrice de sortie
    output = np.zeros( (int(x),int(y)) )
    # on copie l'image a traiter, elle va etre modifiee
    imageCp = deepcopy(image)
    # ajout de zeros autour de l'image depend de l'entier zeroPad
    for k in range(zeroPad):
        imageCp = np.insert(imageCp, imageCp.shape[0], 0, axis = 0)
        imageCp = np.insert(imageCp, imageCp.shape[1], 0, axis = 1)
        imageCp = np.insert(imageCp,0,0, axis = 1)
        imageCp = np.insert(imageCp,0,0, axis = 0)
    # calcul de la sortie
    for i in range(0, imageCp.shape[0]-2*zeroPad, stride):
        for j in range(0, imageCp.shape[1]-2*zeroPad, stride):
            output[i,j] = (np.multiply(imageCp[i: i+len(filterIm[0]), j: j+len(filterIm[1])]  , filterIm)).sum()
    return output

# image en RVB
image = img.imread('city.jpg')
imageRed = image[::,::,0]
imageBlue = image[::,::,1]
imageGreen = image[::,::,2]


figure(1)
# filtre
filt = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
convo = conv(imageBlue, filt, 1,1)
fig = plt.figure(figsize=(image.shape[0],image.shape[1]))

fig.add_subplot(2,2,1)
plt.imshow(image)
fig.add_subplot(2,2,2)
plt.imshow(convo)
plt.show()
