# %load network.py

"""
networksteep.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    

    def __init__(self, sizes, return_vector=False):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.steepener = 0
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.return_vector = return_vector

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b, self.steepener)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None,drop_in=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            
            
      # np.savetxt(ws.txt, self.weights)
      # np.savetxt(bx.txt, self.biases)

        for j in range(epochs):
            
            self.steepener+=1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, drop_in=drop_in)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))
            
      
    def SGD_special(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #WHAT??
        
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            
            
      # np.savetxt(ws.txt, self.weights)
      # np.savetxt(bx.txt, self.biases)
        last_acc = 0.9

        for j in range(epochs):
            
            self.steepener+=1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, drop_in=drop_in)
            if test_data:
                new_acc = self.evaluate(test_data)
                print("Epoch {} : {} / {}".format(j,new_acc,n_test))
                if(last_acc - new_acc > 0.08):
                    self.sizes[1] += 1
                    self.weights[0].append(np.random.rand(784) * 0.001)
                    self.weights[1].append(np.random.rand(784) * 0.001)
                    self.biases[0].append(np.random.rand(784))
                    self.biases[1].append(np.random.rand(784))
                    
                    #then sparsen
                    
                    j -= 1
                    self.steepener -= 1
            else:
                print("Epoch {} complete".format(j))
        
        
    def update_mini_batch(self, mini_batch, eta, drop_in=False, track=None):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if drop_in:
            print(delta_nabla_w[0])
            cutoff = 0.000000001
            print("o", len(self.weights[0][(delta_nabla_w[0] < cutoff) & (delta_nabla_w[0] > -cutoff) & (delta_nabla_w !=0.0)]), "+", len(self.weights[1][(delta_nabla_w[1] < cutoff) & (delta_nabla_w[1] > -cutoff)]))
            self.weights[0][(delta_nabla_w[0] < cutoff) & (delta_nabla_w[0] > -cutoff)] = 0.0
            self.weights[1][(delta_nabla_w[1] < cutoff) & (delta_nabla_w[1] > -cutoff)] = 0.0
        nabla_w[0][self.weights[0]==0.0] = 0.0
        nabla_w[1][self.weights[1]==0.0] = 0.0
    
        if track:
            track[nabla_w > thresh] = 1
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z, self.steepener)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1], self.steepener)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z, self.steepener)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        if self.return_vector:
            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in test_data]

        else:
            test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def copy(net):
        net2 = Network(net.sizes)
        net2.weights = net.weights
        net2.biases = net.biases
        return net2

#### Miscellaneous functions
def sigmoid(z, step):
    """The sigmoid function."""
    if step >= 29:
        #might it make a signficant difference if this was z>=0?
        return 1 * (z > 0)
    return 1.0/(1.0+np.exp(- z * step))

def sigmoid_prime(z,step=1):
    """Derivative of the sigmoid function."""
    return sigmoid(z, step)*(1-sigmoid(z, step))
