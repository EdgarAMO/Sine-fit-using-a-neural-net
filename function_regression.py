# Sine function fit
# 06 / 25 / 2021
# Edgar A. M. O.

from d2l import mxnet as d2l
from mxnet import gluon, autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
import numpy
import math

d2l.use_svg_display()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Auxiliary functions
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def load_array(data_arrays, batch_size, is_train=True): #@save
    """ construct a Gluon data iterator """
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

class Accumulator: #@save
    """ For accumulating sums over `n` variables """
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_loss(net, data_iter, loss): #@save
    """ Evaluate the loss of a model on the given dataset """
    # Sum of losses, no. of examples
    metric = d2l.Accumulator(2) 
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), l.size)
    return metric[0] / metric[1]

def train(net, train_iter, loss, updater): #@save
    """ Train a model within one epoch """
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])

class Animator: #@save
    """ For plotting data in animation. """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Main code
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def plot(x, y, y_hat):
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x, y_hat)
    plt.legend(('Analytical solution', 'Prediction'))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def fit(train_features, test_features, train_labels, test_labels, n=200):
    # loss function
    loss = gluon.loss.L2Loss()
    # net type
    net = nn.Sequential()
    # add layers
    net.add(nn.Dense(100, activation='tanh'))
    net.add(nn.Dense(100, activation='tanh'))
    net.add(nn.Dense(1))
    # initialize weights
    net.initialize()

    batch_size = min(10, train_labels.shape[0])

    train_iter = load_array((train_features, train_labels), batch_size)
    test_iter = load_array((test_features, test_labels), batch_size)

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            yscale='log',
                            xlim=[1, n],
                            ylim=[1e-3, 1e2],
                            legend=['train', 'test'])

    for epoch in range(n):
        train(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_iter, loss),
                         evaluate_loss(net, test_iter, loss)))
            print('epoch ' + str(epoch + 1) + ' completed...')
            
    #print('weight:', net[0].weight.data().asnumpy())
    #print('weight:', net[1].weight.data().asnumpy())
    d2l.plt.show()

    x = np.linspace(a, b, n_train + n_test).reshape(-1, 1)
    
    plot(x, np.sin(x), net(x))

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# This will be executed automatically
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

if __name__ == '__main__' :
    n_train, n_test = 1000, 360             # Training and test dataset sizes

    a = -math.pi                            # feature sample lower limit
    b =  math.pi                            # feature sample upper limit

    # shuffle the sample sample and reshape it as (1360, 1)
    features = (b - a) * np.random.rand(n_train + n_test) + a
    np.random.shuffle(features)
    features = features.reshape(-1, 1)

    # the labels are simply the value of 'y' with added noise.
    labels = np.sin(features)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # train the net:
    X_train = features[:n_train]
    X_test = features[n_train:]

    y_train = labels[:n_train]
    y_test = labels[n_train:]

    fit(X_train, X_test, y_train, y_test)







