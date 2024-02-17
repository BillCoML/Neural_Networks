import numpy as np
from tqdm import trange

class NeuralNetwork:
    def __init__(self):
        #Status of each step, Add Layers must be fixed before Passing in Data
        self.status = False
        #A list of Hidden neurons, ith index is # of Neurons of ith layer
        self.hiddenNeurons = []
        #A list of Hidden Weights and Biases, ith index is Weight matrix to the next layer
        self.weights = []
        self.biases  = []
        #A list of dZ[i]
        self.dZ = []
        #A list of A[i], where A[0] is X(the inputs), A[n] is the outputs
        self.Z = []
        self.A = [] 
        #A list of Activation Functions
        self.activation_functions = []

    def set(self, unit:bool, alpha:float, epoch:int):
        self.unit = unit
        self.alpha = alpha
        self.epoch = epoch

    def ReLU(self, Z):
        return np.maximum(0, Z)
    
    def ReLU_deriv(self, Z):
        return Z > 0
    
    def SoftMax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T
    
    def get_predictions(self, A):
        return np.argmax(A, 0)
    
    def get_accuracy(self, predictions, actual):
        passed = np.sum(predictions == actual)
        accuracy = passed / actual.size
        return accuracy
    
    def forward_propagation(self, i=0):
        if i == len(self.weights)-1:
            #Last Layers
            self.Z[i] = self.weights[i] @ self.A[i]
            self.A[i+1] = self.SoftMax(self.Z[i] + self.biases[i])
            return
        
        self.Z[i] = self.weights[i] @ self.A[i]
        self.A[i+1] = self.ReLU(self.Z[i] + self.biases[i])
        self.forward_propagation(i+1)

    def backward_propagation(self, i):
        dW = self.dZ[i] @ self.A[i].T
        if self.unit:
            dW = dW / (dW**2).sum()**0.5
        self.weights[i] -= self.alpha * dW
        db = 1/self.samples * np.sum(self.dZ[i])
        self.biases[i]  -= self.alpha * db
        if i == 0:
            return
        self.dZ[i-1] = self.weights[i].T @ self.dZ[i] * self.ReLU_deriv(self.Z[i-1])
        self.backward_propagation(i-1) 

    def gradient_descent(self):
        self.one_hot_Y = self.one_hot(self.y_train)
        currentPoch = trange(self.epoch, desc='', leave=True)
        for i in currentPoch:
            self.dZ[-1] = 1/self.samples * (self.A[-1] - self.one_hot_Y)
            self.forward_propagation()
            self.backward_propagation(len(self.dZ)-1)
            if i % 20 == 0:
                predictions = self.get_predictions(self.A[-1])
                accuracy = self.get_accuracy(predictions, self.y_train)
                currentPoch.set_description(f'Accuracy={accuracy*100:.2f}%')
                currentPoch.refresh()

    #Generate Neural network Structure
    def addLayer(self, neurons, activation_function=None):
        if self.status: 
            return
        allowed = ['relu','tanh','sigmoid']
        self.hiddenNeurons.append(neurons)
        if activation_function == None:
            self.status = True
            return
        if activation_function not in allowed:
            print('error_activation function not found')
        else:
            self.activation_functions.append(activation_function)

    def init_params(self):
        self.weights.append(np.random.rand(self.hiddenNeurons[0] , self.n_in) - 0.5)
        self.biases.append(np.random.rand(self.hiddenNeurons[0], 1) - 0.5)
        for n in range(len(self.hiddenNeurons)-1):
            self.weights.append(np.random.rand(self.hiddenNeurons[n+1],self.hiddenNeurons[n]) - 0.5)
            self.biases.append(np.random.rand(self.hiddenNeurons[n+1], 1) - 0.5)
            self.Z.append(0)
            self.dZ.append(0)
            self.A.append(0)
        self.Z.append(0)#offset
        self.dZ.append(0)#offset
        self.A.append(0)#offset
        self.status = True

    def fit(self, x_train, y_train):
        #Must be at least one Hidden layer
        self.A.append(x_train)
        self.y_train = y_train
        m, n = x_train.shape
        self.n_in  = m
        self.samples = n
        self.n_out = self.hiddenNeurons[-1]
        self.init_params()
        self.gradient_descent()

    def evaluate(self, x_test, y_test):
        self.A[0] = x_test
        self.y_train = y_test
        self.forward_propagation()
        predictions = self.get_predictions(self.A[-1])
        accuracy = self.get_accuracy(predictions, y_test)
        print(f'accuracy {(accuracy * 100):.2f} %')