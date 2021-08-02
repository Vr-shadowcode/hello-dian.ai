import numpy as np
from itertools import product
from numpy.core.fromnumeric import transpose

from numpy.random import gamma
import nn.tensor
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            # isintance():判断attr的类型是不是Module
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            # isintance():判断attr的类型是不是Module
            if isinstance(attr, Module):    
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # TODO Initialize the weight
        # of linear module.

        self.w = tensor.from_array(np.random.randn(in_length+1,out_length))
        # self.b = np.zeros(out_length)
        self.x = None
        # self.original_x_shape = None
        # self.dw = None
        # self.db = None

        # End of todo

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.

        
        self.original_x_shape = x.shape
        # print(x.shape)
        v = np.ones(x.shape[0])
        x = np.column_stack([x,v])

        self.x = x
        # print(x.shape)
        # print(self.w.shape)
        out = np.dot(x,self.w) 
        
        return out


        # End of todo


    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        

        # dx = np.dot(dy,self.w.T)
        self.w.grad = np.dot(self.x.T,dy)
        # dx = dx.reshape(*self.original_x_shape)
        ...
        # return dx


        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.
        ...
        self.L = int(length)
        self.running_mean = np.zeros(self.L)  # 追踪mini-batch 均值
        self.running_var = np.ones(self.L)   # 追踪mini-batch 方差
        self.eps = 1e-5  # 排除计算错误和分母为0的情况
        self.momentum = momentum  # 超参数,追踪样本整体均值和方差的动量
        self.beta = nn.tensor.from_array(np.zeros(shape=(self.L,)))
        # self.gamma = gamma(0,1,self.L).reshape((self.L,))
        self.gamma = nn.tensor.from_array(np.ones((self.L)))
        self.x_hat = None
        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.
        ...
        x_mean = np.mean(x,axis=0)
        x_var = np.var(x,axis=0)
        # 根据计算的mean和var批量归一化x
        self.x_hat = (x-x_mean) / np.sqrt(x_var + self.eps)
        y = self.gamma*self.x_hat + self.beta

        # 根据当前mini-batch的样本进行追踪更新，计算滑动平均
        self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*x_mean  
        self.running_var = (1-self.momentum)*self.running_var + self.momentum*x_var

        return y
        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.
        ...
        # 需要保存前向传播里面的部分参数值(中间变量):
        #   self.x_hat:
        #   self.gamma:
        #   x-x_mean:
        #   x_var+self.eps:


        N = self.x_hat.shape[0]
        self.gamma.grad = np.sum(self.x_hat * dy,axis=0) 
        self.beta.grad = np.sum(dy,axis=0)

        dx_hat  = np.matmul(np.ones((N,1)),gamma.reshape((1,-1))) * dy
        dx = N*dy - np.sum(dx_hat,axis=0) - self.x_hat * np.sum(dx_hat*self.x_hat)
        
        return dx


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=False):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.
        ...
        self.C_in = in_channels
        self.C_out = channels
        self.kernel_size = kernel_size
        # self.n_filters = C_out
        self.W = nn.tensor.from_array(np.random.randn(self.C_in,self.C_out,self.kernel_size,self.kernel_size))
        self.W.grad = np.zeros((self.kernel_size,self.kernel_size))
        self.W_im2col = None
        self.b = nn.tensor.from_array(np.zeros((1,self.C_out)))
        self.b.grad = np.zeros((1,self.C_out))
        self.stride = stride
        self.padding = padding
        self.x = None
        self.x_padded = None
        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        ...
        batch_size,C_in,H_in,W_in = x.shape
        self.x = x

        # FC,FN,FH,FW = self.W.shape
        # FC:=C_in, 滤波器的通道数，由输入的x的通道数决定C_in
        # FN:=C_out, 滤波器W的个数,决定输出out的通道数
        # FH:滤波器卷积核的尺寸大小，由自己设计或者采用默认的kernel_size都可以,通常是3
        # FW:滤波器卷积核的尺寸大小，由自己设计或者采用默认的kernel_size都可以,通常是3

        # self.x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding),'constant'))
        # self.W_im2col = self.W.reshape((self.n_filters),-1)
        # self.layer_input = x

        # 调用Conv2d_im2col对x进行拉伸
        # x:(B,C_in,H_in,W_in) -> (B*H_out*W_out,C_in*FH*FW)
        #                      -> (B*H_out*W_out,C_in*kernel_size*kernel_size)      
        x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')                
        self.X_im2col = Conv2d_im2col.forward(self,x_padded)

        # self.W也要进行拉伸，变成二维
        # self.W:(C_in,C_out,self.kernel_size,self.kernel_size) ->  (FH*FW*C_in,FN)
        #                                                       ->  (FH*FW*C_in,C_out)
        self.W_im2col = self.W.reshape(self.C_out,-1).T
        out = np.dot(self.X_im2col,self.W_im2col) + self.b

        self.out = out.reshape(
                batch_size,(H_in+2*self.padding-self.kernel_size) // self.stride + 1,(W_in+2*self.padding-self.kernel_size) // self.stride + 1,-1) \
                .transpose(0,3,1,2)
        
        return self.out
        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        ...
        dy = dy.transpose(0,2,3,1).reshape(-1,self.out_channels)

        self.W.grad = np.dot(self.X_im2col.T,dy).transpose(1,0).reshape(self.W.shape)
        self.b.grad = np.sum(dy,axis=1,keepdims=True)
        dx_im2col = np.dot(dy,self.W_im2col.T)

        # backward时转换col2im
        B,C_in,H_in,W_in = self.x.shape
        out_h = int((H_in+2*self.padding-self.kernel_size) / self.stride - 1)
        out_w = int((W_in+2*self.padding-self.kernel_size) / self.stride - 1)

        dx_im2col = dx_im2col.reshape(B,out_h,out_w,C_in,self.kernel_size,self.kernel_size),transpose(0,3,4,5,1,2)
        dx = np.zeros((B,C_in,H_in+2*self.padding+self.stride-1,W_in+2*self.padding+self.stride-1))
        for x in np.arange(self.kernel_size):
            for y in np.arange(self.kernel_size):
                dx[:,:,x:x+self.stride*out_h:self.stride,y:y+self.stride*out_w] += dx_im2col[:,:,x,y,:,:]

        return dx[:,:,self.padding:H_in+self.padding,self.padding:W_in+self.padding]
        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        ...
        """
        batch_size,C_in,height,width = x.shape
        filter_height,filter_width = self.kernel_size
        # pad_h,pad_w = int((filter_height - 1) / 2),int((filter_width-1) / 2)
        # x_padded = np.pad(x,((0,0),(0,0),pad_h,pad_w),mode='constant')
        out_height = int((height+2*pad_h-filter_height)/ self.stride - 1)
        out_width = int((width+2*pad_w-filter_width)/ self.stride - 1)

        i0 = np.repeat(np.arange((filter_height),filter_width))
        i0 = np.tile(i0,C_in)
        i1 = self.stride * np.repeat(np.arange(out_height),out_width)
        j0 = np.tile(np.arange(filter_width),filter_width * C_in)
        j1 = self.stride * np.tile(np.arange(out_width),out_width)

        i = i0.reshape(1,-1) + i1.reshape(-1,1)
        j = j0.reshape(1,-1) + j1.reshape(-1,1)
        # i.shape:(out_weight*out_width,C_in*filter_height*filter_width),存放的是从x_padded中一个通道每次读取一个单元卷积核对应元素的第一轴的索引位置
        # j.shape:(out_weight*out_width,C_in*filter_height*filter_width),存放的是从x_padded中一个通道每次读取一个单元卷积核对应元素的第二轴的索引位置

        k = np.repeat(np.arange(C_in),filter_width*filter_height).reshape(1,-1)
        # k.reshape:(1,C_in*filter_height*filter_width),存放的是每次卷积通道的索引

        im_col = x_padded[:,k,i,j]
        # 对x_padded进行切片处理,[:,k,i,j]就是根据k,i,j索引提取出x_padded中需要卷积的元素
        # imcol.shape:(batch_size,out_weight*out_width,C_in*filter_height*filter_width)

        # 用transpose对imcol维度变换之后让每一行的元素个数是C_in*filter_width*filter_height,
        # 也就是每一次卷积需要卷积的元素个数，对每个样本进行卷积,包括所有通道
        # im_col = im_col.transpose(1,2,0).reshape(-1,filter_width*filter_height*C_in)
        im_col = im_col.reshape(-1,filter_width*filter_height*C_in)

        return im_col
        """
        B,iC,iH,iW = x.shape
        p,s,k = self.padding,self.stride,self.kernel_size
        oH,oW = (iH-k) // s + 1,(iW-k) // s + 1
        col = np.zeros((B,iC,k,k,oH,oW))
        for h in np.arange(k):
            for w in np.arange(k):
                col[:,:,h,w,:,:] = x[:,:,h:h+s*oH:s,w:w+s*oW:s]
        return col.transpose(0,4,5,1,2,3).reshape(B*oH*oW,-1)

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.
        ...
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.x_padded = None
        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.
        ...
        B,C,H_in,W_in = x.shape
        # self.x = x
        self.x_padded = np.pad(x,((0,0),(0,0),(self.padding),(self.padding)),mode = 'constant')
        out_h = int((H_in+2*self.padding-self.kernel_size) / self.stride + 1)
        out_w = int((W_in+2*self.padding-self.kernel_size) / self.stride + 1)
        self.out = np.zeros(B,C,out_h,out_w)

        for h in np.arange(out_h):
            for w in np.arange(out_w):
                self.out[:,:,h,w] = np.mean(self.x_padded[:,:  
                        self.stride*h:self.stride*h+self.kernel_size,self.stride*w:self.stride*w+self.kernel_size],axis=(-2,-1))

        return self.out
        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.
        ...
        B,C,H_out,W_out = dy.shape
        B,C,H_in,W_in = self.x.shape
        self.grad = np.zeros_like(self.x_padded)
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                self.grad[:,:,self.stride*h:self.stride*h+self.kernel_size,self.stride*w:self.stride*w+self.kernel_size] \
                    += dy[:,:,h,w] / self.kernel_size**2

        return self.grad[:,:,self.padding:self.padding+H_in,self.padding:self.padding+W_in]
        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        ...
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.x_padded = None
        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        ...
        B,C,H_in,W_in = x.shape
        self.x = x
        x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')

        out_h = int((H_in+2*self.padding-self.kernel_size) / self.stride + 1)
        out_w = int((W_in+2*self.padding-self.kernel_size) / self.stride + 1)
        self.out = np.zeros((B,C,out_h,out_w))
        for h in np.arange(out_h):
            for w in np.arange(out_w):
                self.out[:,:,h,w] = np.max(x_padded[:,:,
                self.stride*h:self.stride*h+self.kernel_size,self.stride*w:self.stride*w+self.kernel_size],axis=(-2,-1))

        return self.out
        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.
        ...
        B,C,H_out,W_out = dy.shape
        B,C,H_in,W_in = self.x.shape
        self.grad = np.zeros_like(self.x_padded)
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                tmp_x_padded = self.x_padded[:,:,self.stride*h:self.stride*h+self.kernel_size,self.stride*w:self.stride*w+self.kernel_size]
                id_max = np.max(tmp_x_padded,axis=(-2,-1))
                tmp_grad = self.grad[:,:,self.stride*h:self.stride*h+self.kernel_size,self.stride*w:self.stride*w+self.kernel_size]
                tmp_grad += np.where(tmp_grad==np.expand_dims(id_max,(-2,-1)),dy[:,:,h,w],0)

        return self.grad[:,:,self.padding:self.padding+H_in,self.padding:self.padding+W_in]
        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.
        ...
        self.p = p
        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.
        ...
        a = (np.random.rand(*x.shape) < self.p) 
        a = a / self.p

        return a*x
        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        ...

        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()