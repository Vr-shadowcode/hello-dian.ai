import numpy as np
from .modules import Module


class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        ...
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        ...
        dx = dy * self.out * (1-self.out)
        return dx
        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.
        ...
        out = 2 / (1+np.exp(-2*x)) - 1
        self.out = out
        return out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.
        ...
        dx = (1-self.out*self.out) * dy
        return dx
        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        ...
        self.mask = (x <= 0)
        out = x.copy
        # 将x<0的初始化为0
        out[self.mask] = 0
        return out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        ...
        # 小于0的位置，反向传播的误差为0
        dy[self.mask] = 0
        dx = dy
        return dx
        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        ...
        if x.ndim == 2:
            x = x - np.max(x,axis=1)
            self.out = np.exp(x) / np.sum(np.exp(x),axis=1)
            return self.out

        x = x - np.max(x)
        self.out = np.exp(x) / np.sum(np.exp(x))

        return self.out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        ...
        dx = (np.diag(self.out)-np.outer(self.out,self.out)) * dy
        return dx
        # End of todo


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        ...
        self.probs = Softmax(probs)
        self.targets = targets
        self.loss = CrossEntropyLoss(self.probs,self.targets)
        # self.dp = None
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.
        ...
        batch_size = self.targets.shape[0]
        if self.targets.size == self.probs.size:
            dx = (self.probs - self.targets) / batch_size

        else:
            dx = self.probs.copy()
            dx[np.arange(batch_size),self.targets] -= 1
            dx = dx / batch_size

        return dx
        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.
        self.probs = Softmax(probs)
        if probs.ndim == 1:
            self.probs = probs.reshape(1,probs.size)
            self.targets = targets.reshape(1,targets.size)

        if probs.size == targets.size:
            self.targets = targets.argmax(axis=1)

        batch_size = self.probs.shape[0]

        self.loss = -np.sum(np.log(self.probs[np.arange(batch_size),targets] + 1e-7)) / batch_size

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        ...
        # dx = - (self.targets / self.probs)
        batch_size = self.targets.shape[0]

        if self.targets.size == self.probs.size:
            dx = (self.probs - self.targets) / batch_size

        else:
            dx = self.probs.copy()
            dx[np.arange(batch_size,self.targets)] -= 1
            dx = dx / batch_size
        return dx
        # End of todo
