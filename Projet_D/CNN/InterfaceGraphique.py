from tkinter import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk
import time
import Perceptron as p
import imageReader
import pickle
import inmageTest

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(file):
    labelled_images = unpickle(file)
    print('getting data from file', file, '... ', end = '')
    X = np.asarray(labelled_images[b'data']).astype("uint8")
    X = np.reshape(X, (10000,3,32,32))
    # X = X.transpose([0, 2, 3, 1])
    X = X/float(255)
    Yraw = np.asarray(labelled_images[b'labels'])
    Y = np.zeros((10000,10))
    for i in range(10000):
        Y[i,Yraw[i]] = 1
    print('done')
    return X,Y

def imgSave(dataImg, numImg):
    transitimg = np.rollaxis(dataImg[0][numImg], 0, 3) * 255
    img = Image.fromarray(transitimg.astype("uint8"))
    img = img.resize((32*5,32*5))
    img.save('my.png')

def switch_class(number):
    switcher = {
        0 : "Plane",
        1 : "Car",
        2 : "Bird",
        3 : "Cat",
        4 : "Deer",
        5 : "Dog",
        6 : "Frog",
        7 : "Horse",
        8 : "Ship",
        9 : "Truck"
    }
    return switcher.get(number)


def testWithInterface(inFile, test_images, netScratch):
    fenetre = Tk()
    canvas = Canvas(fenetre, width=1350, height=800, background="black")
    canvas.pack()

    for i in range(10):
        txt = canvas.create_text(75, i*65 +110, text=switch_class(i), font="Arial 25", fill="white")

    txt = canvas.create_text(200, 40, text="Scratch", font="Arial 40 bold", fill="yellow")

    for i in range(10):
        a2 = canvas.create_rectangle(130,i*65 + (65/2 - 5) +60,325, i*65 +3*65/2-25 + 60, outline = 'white')

    for training in range(len(test_images[0])):
        training = np.random.randint(len(test_images[0]))
        imgSave(test_images, training)
        new_img = PhotoImage(file = 'my.png')
        image = canvas.create_image(650,350, image = new_img)
        label = canvas.create_text(650, 500, text=switch_class(np.argmax(test_images[1][training])), font = "Arial 35", fill = 'pink')
        canvas.update()
        input()
        resultatScratch = netScratch.propagation(test_images[0][training])
        for j in range(10):
            remplissage = canvas.create_rectangle(130,j*65 + 60 + (65/2 - 5), 130+195*resultatScratch[j], j*65 + 60 +3*65/2-25, fill ='white', outline = 'white')
            canvas.update()
        input()
        sScratch = np.argmax(resultatScratch)
        if sScratch == np.argmax(test_images[1][training]):
            colorScratch = 'green'
        else :
            colorScratch = 'red'

        sortieScratch = canvas.create_rectangle(130,sScratch*65 + 60 + (65/2 - 5), 130+195*resultatScratch[sScratch], sScratch*65 + 60 +3*65/2-25, fill =colorScratch, outline = 'white')
        canvas.update()
        input()

        for i in range(10):
            a2 = canvas.create_rectangle(130,i*65 + (65/2 - 5) +60,325, i*65 +3*65/2-25 + 60, fill = 'black', outline = 'white')
        canvas.delete(image)
        canvas.delete(label)
        canvas.delete(sortieScratch)

    fenetre.mainloop()

def imgSaveMnist(inFile, numImg):
    data = inmageTest.convert_matrix(inFile, 16 + 784 * numImg).astype('uint8')
    data = np.fliplr(data)
    img = Image.fromarray(data)
    img = img.resize((28*20,28*20))
    img = img.rotate(90)
    img.save('my.png')


def interfaceEMnist(Perceptron, inFile, cas, test_images):
    if cas == True :
        var = 25
    else :
        var = 65
    fenetre = Tk()
    canvas = Canvas(fenetre, width=1350, height=700, background="black")
    canvas.pack()

    for i in range(int(10*65/var)):
        if(cas) :
            txt = canvas.create_text(900, i*var +50*var/65, text=chr(i+97), font="Arial " + str(int(35*var/65)), fill="white")
        else :
            txt = canvas.create_text(900, i*var +50*var/65, text=str(i), font="Arial " + str(int(35*var/65)), fill="white")
        a2 = canvas.create_rectangle(930,i*var + (var/2 - 5*var/65),1125, i*var +3*var/2-25*var/65, outline = 'white')


    for i in range(10000):
        i = np.random.randint(10000)
        imgSaveMnist(inFile,i)
        new_img = PhotoImage(file = 'my.png')
        image = canvas.create_image(400,350, image = new_img)
        canvas.update()
        input()
        resultat = Perceptron.propagation(test_images[0][i])
        for j in range(len(resultat)):
                remplissage = canvas.create_rectangle(930,j*var + (var/2 - 5*var/65), 930+195*resultat[j], j*var +3*var/2-25*var/65, fill ='white', outline = 'white')
        canvas.update()
        input()
        s = np.argmax(resultat)
        if s == np.argmax(test_images[1][i]):
            sortie = canvas.create_rectangle(930,s*var + (var/2 - 5*var/65), 930+195*resultat[s], s*var +3*var/2-25*var/65, fill ='green', outline = 'white')
        else :
            sortie = canvas.create_rectangle(930,s*var + (var/2 - 5*var/65), 930+195*resultat[s], s*var +3*var/2-25*var/65, fill ='red', outline = 'white')
        canvas.update()
        input()
        canvas.delete(sortie)
        for i in range(int(10*65/var)):
            a2 = canvas.create_rectangle(930,i*var + (var/2 - 5*var/65),1125, i*var +3*var/2-25*var/65,fill = 'black', outline = 'white')
        canvas.delete(image)
        canvas.update()
        input()


    fenetre.mainloop()

#test_images = inmageTest.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0)
#interfaceMnist(percep,'t10k-images-idx3-ubyte', True, 10000, test_images, labelled_images,60000)
i=input()
if(int(i)==1):
##################### Letters test ##########################
    test_images = imageReader.list_labelled_images2D('emnist-letters-test-images-idx3-ubyte', 'emnist-letters-test-labels-idx1-ubyte', 10000, 0, 'letters')
    interfaceEMnist(pickle.load(open('Tests/MNIST/EMnist_CNN_batch4','rb'))[0][1],'emnist-letters-test-images-idx3-ubyte',True, test_images)

else :
##################### Cifar Test ###########################
    test_images = get_data('cifar-10-batches-py/test_batch')
    testWithInterface('cifar-10-batches-py/test_batch', test_images, pickle.load(open('cifar-10-batches-py/CIFAR10_1batch5', 'rb'))[0][1])
