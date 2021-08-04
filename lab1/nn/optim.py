from .tensor import Tensor
from .modules import Module

# 几类优化器的实现
class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.
        ...  
        # 反向传播看作一张图，每层的参数都会进行反向传播求出每层参数的梯度,
        # 调用optim可以进行梯度优化算法将梯度用于计算后每层的参数都会更新的新的值
        if isinstance(module,Module):
            for i in range(len(module.layer)):
                self._update_weight(module.layer[i].tensor)
        elif isinstance(module,Tensor):
            self._update_weight(module)
        elif isinstance(module,list):
            for i in range(len(module)):
                self._step_module(module[i])
        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim): # 随机梯度下降

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum
        self.tensor_dt = None

    def _update_weight(self, tensor):

        # TODO Update the weight of tensor
        # in SGD manner.
        ...
        if self.tensor_dt is None:
            self.tensor_dt = Tensor.from_array(tensor.grad)
        self.tensor_dt = self.momentum*self.tensor_dt + (1-self.momentum)*tensor.grad

        tensor -= self.lr*self.tensor_dt
        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.
        ...
        self.beta = [0.99,0.999]
        self.eps = 1e-7
        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.
        ...
        
        # End of todo
