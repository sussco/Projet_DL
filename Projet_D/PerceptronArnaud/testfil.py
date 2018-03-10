import pickle

with open("test1","r") as f:
    while 1:
        try:
            x = pickle.load(f)
            print x
        except EOFError:
            break
