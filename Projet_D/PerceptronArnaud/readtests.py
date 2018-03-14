import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

listTest = []
for n in sys.argv[1:]:
    print(n) #print out the filename we are currently processing
    with open(n,"rb") as f:
        x = pickle.load(f)
        print (x)
        listTest.append(x)
learningR = []
successR = []
#for i in range(len(listTest)):
for j in range(len(listTest[0][0::3])):
    learningR.append(listTest[0][1::3][j])
    successR.append(listTest[0][2::3][j] )
        #print (listTest)
            #print (a)
for i in range(2):
    for j in range(len(listTest[i+1][0::4])):
        learningR.append(listTest[i+1][1::4][j])
        successR.append(listTest[i+1][3::4][j] )



LRRes= open("list1LR","ab")
pickle.dump(learningR,LRRes)
SRRes= open("list1SR","ab")
pickle.dump(successR,SRRes)
