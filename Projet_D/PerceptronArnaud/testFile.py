

f= open("test.txt","a+")
a = [1,2,3]
for item in a:
    f.write("this is line %d \n" % item)
