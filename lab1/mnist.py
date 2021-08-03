import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from nn.modules import Linear,BatchNorm1d,Conv2d,MaxPool
from nn.functional import ReLU,Sigmoid
# from typing import OrderedDict

import nn
import nn.functional as F

n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
lr = 1e-3
lengths = (n_features, 512, n_classes)


class Model(nn.Module):

    # TODO Design the classifier.
    ...
    def __init__(self,shape):
        self.shape = list(shape)
        self.original_shape = (np.sqrt(n_features),np.sqrt(n_features))  # (28,28)
        self.BN = True
        self.Conv2d = True
        self.pool = MaxPool()
        self.dropout = nn.modules.Dropout()

        if self.Conv2d is True:
            self.convlayer = [BatchNorm1d(int(self.original_shape[0]*self.original_shape[1])),
                            Conv2d(1,1),  # 初始化输入是单通道，卷积核个数是1
                            ReLU(),
                            MaxPool()]   
            self.layer = [Linear(169,512),
                          nn.functional.Sigmoid(),
                          Linear(512,10),
                          nn.functional.Sigmoid()]
        else:
            self.layer = [Linear(shape[0],shape[1]),nn.functional.Sigmoid(),
                         Linear(shape[1],shape[2]),nn.functional.Sigmoid(),
                         ]
        # self.update_num = [0,1,1]

        
    def forward(self,x: np.ndarray) -> np.ndarray: 
        print(x.shape)
        self.input = x
        x = np.expand_dims(self.convlayer[0].forward(x).reshape(-1,28,28),1)
        print(x.shape)
        for i in range(1,len(self.convlayer)):
            x = self.convlayer[i].forward(x)
            print(x.shape)
        x = np.squeeze(x.reshape(x.shape[0]*x.shape[1],-1))

        for layer in self.layer:
                x = layer.forward(x)
        x = self.dropout(x)

        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # for layer in self.layer:
        #     layer.tensor.grad = np.zeros(layer.tensor.shape)

        if self.Conv2d:
            layer = [self.convlayer,self.layer]
        else:
            layer = self.layer

        for i in range(2,len(layer[-2]) + 1):
            dout = self.layer[-i].backward(dy)
        
        for i in range(2,len(layer)+1):
            for j in range(1,len(layer[-i]) + 1):
                dout = layer[-i][-j].backward(dout)

        return dout
    # End of todo
# lab1\train-images.idx3-ubyte

def load_mnist(mode='train', n_samples=None):
    images = 'lab1/train-images.idx3-ubyte' if mode == 'train' else 'lab1/t10k-images.idx3-ubyte'
    labels = 'lab1/train-labels.idx1-ubyte' if mode == 'train' else 'lab1/t10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape((length, 28, 28)).astype(np.int32)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape((length)).astype(np.int32)
    return (X[:n_samples].reshape(n_samples, -1), y[:n_samples]) if n_samples is not None else (X.reshape(length, -1), y)

def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = Model(lengths)
    optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            loss = criterion(probs, y)
            model.backward(loss.backward())
            optimizer.step()
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f} loss={loss.value:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()