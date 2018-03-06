from Tkinter import *
import Image
import numpy as np
import matplotlib.pyplot as plt
import inmageTest
from PIL import ImageTk
import time
import Perceptron as p




def imgSave(inFile, numImg):
    data = inmageTest.convert_matrix(inFile, 16 + 784 * numImg).astype('uint8')
    img = Image.fromarray(data)
    img = img.resize((28*20,28*20))
    img.save('my.png')


def testWithInterface(Perceptron, inFile, cas, rangeTest, test_images, labelled_images, rangeLabel):
    if cas == True :
        var = 25
    else :
        var = 65
    # for i in range(rangeLabel):
    #     percep.propagation(labelled_images[0][i])
    #     percep.backPropagation(labelled_images[1][i])
    #     percep.updateParams(1)
    #     if ((i *100) % rangeLabel ==0):
    #         print("\n \n \n\n\n\n\n\n\n\n\n \n \n\n\n\n\n\n\n\n\n \n \n\n\n\n\n\n\n\n" + str(int((float(i)*100)/float(rangeLabel))) +"%")
    fenetre = Tk()
    canvas = Canvas(fenetre, width=1350, height=700, background="black")
    canvas.pack()

    for i in range(10*65/var):
        if(cas) :
            txt = canvas.create_text(900, i*var +50*var/65, text=chr(i+97), font="Arial " + str(35*var/65), fill="white")
        else :
            txt = canvas.create_text(900, i*var +50*var/65, text=str(i), font="Arial " + str(35*var/65), fill="white")
        a2 = canvas.create_rectangle(930,i*var + (var/2 - 5*var/65),1125, i*var +3*var/2-25*var/65, outline = 'white')


    for i in range(rangeTest):
        imgSave(inFile,i)
        new_img = PhotoImage(file = 'my.png')
        image = canvas.create_image(400,350, image = new_img)
        canvas.update()
        time.sleep(0.5)
        resultat = percep.propagation(test_images[0][i])
        for j in range(len(resultat)):
                remplissage = canvas.create_rectangle(930,j*var + (var/2 - 5*var/65), 930+195*resultat[j], j*var +3*var/2-25*var/65, fill ='white', outline = 'white')
        canvas.update()
        time.sleep(0.6)
        s = np.argmax(resultat)
        sortie = canvas.create_rectangle(930 - 10*var/65,s*var + var/2 - 15*var/65 ,1125 + 10*var/65 , s*var +3*var/2-15*var/65, outline = 'red', width = 3)
        canvas.update()
        time.sleep(0.5)
        canvas.delete(sortie)
        for i in range(10*65/var):
            a2 = canvas.create_rectangle(930,i*var + (var/2 - 5*var/65),1125, i*var +3*var/2-25*var/65,fill = 'black', outline = 'white')
        canvas.delete(image)
        canvas.update()
        time.sleep(0.5)


    fenetre.mainloop()

labelled_images = inmageTest.list_labelled_images('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 60000, 0)
test_images = inmageTest.list_labelled_images('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0)
percep = p.Perceptron([784,300,100,10],0.8,0)
testWithInterface(percep,'t10k-images-idx3-ubyte', True, 10000, test_images, labelled_images,60000)
