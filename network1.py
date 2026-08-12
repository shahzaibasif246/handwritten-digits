import random
import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    sp = (sigmoid(z)) * (1-sigmoid(z))
    return sp
    
class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x,y in zip(sizes[1:], sizes[:-1])]
        self.loss = list()

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, predicted, output):
        return (predicted - output)
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def evaluate(self, test_data):
        test_results = list()
        
        loss_before_avg = list()
        for x, y in test_data:
            activation = self.feedforward(x)
            test_results.append((np.argmax(activation),y))
            loss = self.loss_func(activation, y)
            loss_before_avg.append(loss)
        
        epoch_loss = sum(loss_before_avg)/len(loss_before_avg) * 0.5
        self.loss.append(epoch_loss)
        #test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        total_correct = 0
        for x,y in test_results:
            if x == y:
                total_correct +=1
        return total_correct
    def loss_func(self, activation, y):
        y_vec = np.zeros_like(activation)
        y_vec[y] = 1.0
        diff = y_vec-activation
        loss = np.linalg.norm(diff) ** 2
        return loss





