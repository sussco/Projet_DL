from perceptron import *

#An example
perceptronANDgate = Perceptron(0, 0, 1, 2, 0.05, 0.5, "trigger") #2 inputs excluding BIAS TODO : try changing learning rate and relative importance
perceptronANDgate.printPerceptron()
training_in = [
    [[0], [0]], #try 1.
    [[0], [1]],
    [[1],[0]],
    [[1], [1]]]
training_out = [[0], [0], [0], [1]]
#Init the perceptron
print("\n ----INIT----")

perceptronANDgate.initW_b() #essayer avec une loi normale comme dans l'exemple
perceptronANDgate.printPerceptron()

#Propagation first example
print("\n---------EXAMPLE 1-------")
perceptronANDgate.forwardPropagation(training_in[0])
perceptronANDgate.printPerceptron()
print("\nLearning :")
perceptronANDgate.learning(training_out[0])
perceptronANDgate.forwardPropagation(training_in[0])
perceptronANDgate.printPerceptron()

#Propagation second example
print("\n-----EXAMPLE 2-----")
perceptronANDgate.forwardPropagation(training_in[1])
perceptronANDgate.printPerceptron()
print("\nLearning :")
#perceptronANDgate.learning(training_out[1])
perceptronANDgate.forwardPropagation(training_in[1])
perceptronANDgate.printPerceptron()

#Propagation third example
print("\n-----EXAMPLE 3-----")
perceptronANDgate.forwardPropagation(training_in[2])
perceptronANDgate.printPerceptron()
print("\nLearning :")
#perceptronANDgate.learning(training_out[2])
perceptronANDgate.forwardPropagation(training_in[2])
perceptronANDgate.printPerceptron()

#Propagation FOURTH example
print("\n-----EXAMPLE 4-----")
perceptronANDgate.forwardPropagation(training_in[3])
perceptronANDgate.printPerceptron()
print("\nLearning :")
perceptronANDgate.learning(training_out[3])
perceptronANDgate.forwardPropagation(training_in[3])
perceptronANDgate.printPerceptron()

print("\n-----EXAMPLE 2 learn again-----")
perceptronANDgate.forwardPropagation(training_in[1])
perceptronANDgate.printPerceptron()
print("\nLearning :")
#perceptronANDgate.learning(training_out[1])
perceptronANDgate.forwardPropagation(training_in[1])
perceptronANDgate.printPerceptron()

print("\n-----ALL EXAMPLES test-----")
perceptronANDgate.forwardPropagation(training_in[0])
perceptronANDgate.printPerceptron()
perceptronANDgate.forwardPropagation(training_in[1])
perceptronANDgate.printPerceptron()
perceptronANDgate.forwardPropagation(training_in[2])
perceptronANDgate.printPerceptron()
perceptronANDgate.forwardPropagation(training_in[3])
perceptronANDgate.printPerceptron()