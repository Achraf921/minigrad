#let's match the PyTorch's API on the Neural network module now
import random

#Neuron
class Neuron:
    def __init__(self, nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))
    def __call__(self,x):
        #w*x+b, dot product of w and x vectors, which is then reduced and added the bias
        #x needs to be a vector of the same dimension as self.w obviously
        acc = sum((wi*xi for wi,xi in zip(self.w,x)), self.b) ##reducing as we compute the dot product
        return acc.tanh()
    def parameters(self):
        return self.w + [self.b] #concat list of weights and biases
#Layer
class Layer:
    def __init__(self, ndim, nnum):
        self.neurons = [Neuron(ndim) for _ in range(nnum)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


#Multi layer perceptron (MLP)
class MLP:
    def __init__(self, nin, nout):#nout is a list, it represent the number of neurons at layer index i
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nout))] #array of layers, each of size nout[i] and of dim   

    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        return x #i.e recursively, we have base input x, which gets transformed by layer 0, and the output o that transfo feeds layer 1 all the way till layer nout.len which is the output 
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    def train(self, iter, step, data, expec):
        for k in range(iter):
            #first we do a forward pass
            ypred=[self(x) for x in data]

            #then we evaluate the loss on that forward pass
            loss = sum([(yout-ygt)**2 for ygt, yout in zip(expec,ypred)]) #    Σ(predicted-expected)^2

            #then we compute the gradients to nudge (zero-out (since +=) + backward pass )
            for p in self.parameters():
                p.grad = 0.0 ##re-setting fresh gradient values as gradients are +=
            loss.backward() ## re-computing gradients before using them for parameter nudge (backward pass)
    
            #then we nudge the weights proportionally to their accurate re-computed gradients
            for p in self.parameters():
                p.data+= -step* p.grad #moving weights proportionally to their gradients (if gradient is negative, we add so that it pulls loss down,if positive we decrease to pull loss down as well rather than up)
            #the learning rate here, is a crucial part of training
            #then we print loss
            print(loss)
        print([val.data for val in ypred])#print new model prediction
        print(expec)#print expectations
