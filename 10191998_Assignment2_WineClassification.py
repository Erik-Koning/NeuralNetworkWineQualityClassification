'''
Assignment 2: Backpropogation in Neural Networks
Erik Koning
10191998

This file is commented thouroughly in the hopes someone with minimal
Artificial Neural Network experience can understand exactly what each
value does. And also for my sanity, so i can stay on track.
'''
import math
import random
import os 

random.seed(0)

def importData(filename):
    firstLine = 0;
    file = open(filename, 'r')
    temp = []
    datasetString = []
    dataTitle = []
    for line in file:   
        if firstLine == 0:  
            dataTitle = line.split(",")     
            firstLine += 1
        else:
            temp = line.split(',')                  #build becomes an array of the parsed data
            temp[len(temp)-1] = temp[len(temp)-1].replace('"','').replace('\n','')
            datasetString.append(temp)

    # Convert text to numeric
    '''
    stuff = [[None]*3]*10     WRONG, will reference it self
            |
            |
            v
    stuff = [[None]*3 for i in xrange(10)]      Correct
    '''
    #https://stackoverflow.com/questions/21764049/python-list-writing-through-loop-all-previous-values-being-overwritten
    datasetFloat = []
    datasetFloat = [[0.0]*len(datasetString[0]) for i in range(len(datasetString))]

    lengthDataset = len(datasetString)   #each pattern in the set
    i = 0
    while i < lengthDataset:
        j = 0
        while j < len(temp):                 #Look at the features then the result
            datasetFloat[i][j] = float(datasetString[i][j])
            j += 1
        i+= 1
    file.close()
    #returns arrays

    return datasetFloat, dataTitle


# calculating a random number between a and b
def rand(a, b):
    return (b-a)*random.random() + a

# Making a matrix
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# Sigmoid function tahn
def sigmoid(x):
    return math.tanh(x)

# derivative of sigmoid function
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)     #set weights from inputs to hidden to be random value, some need to be negative to ensure value of a low value are possible
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)     #set weights from hidden to output to a random value, again some need to be negative

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)      #2d array default zero
        self.co = makeMatrix(self.nh, self.no)      #2d array default zero


    #update mehtod returns the resultant output node value(s) as an array given the input value array arguments
    def update(self, inputs):
        #checking to see if NN is compatible with number of inputs
        if len(inputs) != self.ni-1:                        
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):              #if not minus 1 -> ERROR: index out of range
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]                         

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            #address each ith input which has a weight to this jth hidden node
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]    #summation of all inputs * respective weights going into jth HIDDEN node
            self.ah[j] = sigmoid(sum)                     #Is this hidden node activated? sigmoid returns a double. signmoid(sum) -> 1 as sum-> infinity

        # output activations
        for k in range(self.no):
            sum = 0.0
            #address each jth hidden node which has a weight to this kth output node
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]    #summation of all inputs * respective weights going into kth OUTPUT node
            self.ao[k] = sigmoid(sum)                     #what is the sigmoid of the summation of inputs*weights

        #returns the entire activation array of outputs
        return self.ao[:]                                   

    #this function will first be calculating the error of the output nodes by using gradient descent.
    #the derivative of the output function(sigmoid) is passed the actual of the output node, and then 
    #multiplied by the error of the output(desired - actual result)
    #The output delta of the higher layer will be used and propogated down(ie, delta of output nodes to be used in delta of hidden node calculations)
    #smaller the error, the smaller the respectivelayer_delta. read delta as the change required, if delta -> 0 not much change desired
    def backPropagate(self, targets, Lr, M):
        #ANN not compatible with number of targets expected
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        #ERROR TERMS FOR OUTPUT
        output_deltas = [0.0] * self.no                             #initialize all the output deltas to zero
        for k in range(self.no):
            #lets look at each output node
            error = targets[k]-self.ao[k]                            #desired - actual result
            output_deltas[k] = dsigmoid(self.ao[k]) * error         #derivative of the result * error(desired-actual resul)

        #ERROR TERMS FOR HIDDEN
        hidden_deltas = [0.0] * self.nh                             #initialize all deltas to be default zero, delta for each hidden node
        #there are two for loops in this case because each hidden node has a weight to EACH output node. nh * no = numWeight connections from hidden layer to output layer
        for j in range(self.nh):
            error = 0.0
            #for each weight from the jth hidden node to the output nodes we are going to add up the summation error
            #as how well each kth output node "did" as recorded by the(output_deltas[kth]) which was calculated a few lines up, scaled by the weight from the jth node
            for k in range(self.no):
                error += output_deltas[k]*self.wo[j][k]             #+= is the same as error = error +......
            #now that we have the total representative error for the jth hidden node
            #we can take the gradient descent and store it for further propogation. gradient descent will be calced by sending the activation of the jth hidden node to
            #the derivative of the activation function, scaled by the total error that hiddent node was responsible for
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error       

        #update output weights
        for j in range(self.nh):                                    
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]                        #change variable = the delta scalar val * the hidden node activation val
                self.wo[j][k] += Lr*change + M*self.co[j][k]                #new weight = weight + outputnode_delta*activationValue*Lr + momentumVal*previousChangeVal
                self.co[j][k] = change                                      #save change for next iteration

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]                        #change variable = the delta scalar val * the input node activation val
                self.wi[i][j] += Lr*change + M*self.ci[i][j]                #new weight = weight + hiddenNode_delta*activationValue*Lr + momentumVal*previousChangeVal
                self.ci[i][j] = change                                      #save change for next iteration

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*(targets[k]-self.ao[k])**2                         #this is a derived equation for error
        return error


    def test(self, patterns):
        for p in patterns:
            print("Input nodes: ", p[0:len(p)-1], '->', "output nodes: ", self.update(p[0:len(p)-1]))     #displays the input pattern, and then the activation output result output nodes

    #print the weights, nested loop to display each node to node combination value
    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            for j in range(self.nh):
                print(i," input node to ",j, " hidden node ",self.wi[i][j])
        print('Output weights:')
        for j in range(self.nh):
            for k in range(self.no):
                print(j," hidden node to ",k, " output node ",self.wo[j][k])

    def train(self, patterns, iterations=500, Lr=0.4, M=0.1):
        # Lr: learning rate, to monitor how much
        # M: momentum factor, so it does not get stuck in a local min
        for i in range(iterations):
            error = 0.0
            for p in patterns:          #patterns contains all data, The word "Patterns" represents all the individual patterns/data points.
                                        #we use the word data point as in a particular classes identifier, and all of its features.
                inputs = p[0:len(p)-1]           #inputs contain the input values to each input node

                if p[len(p)-1] == 5:
                    targets = [0.95,0.05,0.05]
                elif p[len(p)-1] == 7:
                    targets = [0.05,0.95,0.05]
                elif p[len(p)-1] == 8:
                    targets = [0.05,0.05,0.95]
                else:
                    print("Error reading target value")

                self.update(inputs)     #determine what the activations of the output nodes are provided this particular Pattern (singlular!)
                error = error + self.backPropagate(targets, Lr, M)
            #if i % 100 == 0:
            print('error %-.5f' % error)


def demo():
    # Teach network XOR function
    filename = "assignment2data.csv"
    print("current working directory: ",os.path.dirname(os.path.realpath(__file__)))
    print("importing data from file: ",filename)
    train_data, dataLabels = importData(filename)
    print("import complete, training ANN")

    # create a network with input,hidden,output nodes
    n = NN(len(train_data[0])-1,8, 3)                                       #length -1 inputs becuase last element is target, hidden nodes, and 3 outputs becuase 3 different expected results
    
    # train it with some patterns
    n.train(train_data)
    # test it
    print("\n\n\nTesting")
    n.test(train_data[0:251])

    print("[0.95,0.05,0.05] represents 5\n[0.05,0.95,0.05] represents 7\n[0.05,0.05,0.95] represents 8")

    print("\n\nThe weights are: ")
    n.weights()

if __name__ == '__main__':
    demo()