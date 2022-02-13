"""
Spring 2022, 10-707
Assignment-1
Problem 5: CNN
TA in charge: Tiancheng Zhao, soyeonmin

IMPORTANT:
    DO NOT change any function signatures

    Some modules in Problem 4 like ReLU and LinearLayer are similar to Problem1
    but not exactly same. Read their commented instructions carefully.

Feb 2022
"""

import numpy as np
import copy

# When you submit the code for autograder, comment the load cifar 10 dataset command.
# This is only for experiment.
from load_cifar import trainX, trainy, testX, testy


def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    
    X_padding = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = X.shape
    #print("This is X",X_padding)
    out_h = int((H + 2 * padding - k_height) / stride) + 1
    out_w = int((W + 2 * padding - k_width) / stride) + 1
    #print(out_h,out_w)
    level1 = np.repeat(np.arange(k_height),k_width)
    level1 = np.tile(level1, C)
    everyLevels = 1 * np.repeat(np.arange(out_h),out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    
    slide1 = np.tile(np.arange(k_width), k_height)    
    slide1 = np.tile(slide1, C)
    everySlides = 1 * np.tile(np.arange(out_w),out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(C), k_height*k_width).reshape(-1, 1)
    
    cols = X_padding[:,d,i,j]
    #cols = np.concatenate(cols,axis=1)
    cols = cols.transpose(1, 2, 0).reshape(C*k_height*k_width, -1)
    #print("This is cols shape:",cols[3])
    #imm=np.array(cols)          
    return cols
    

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    N, C, H, W = X_shape
    #print("This is X",X_padding)
    out_h = int((H + 2 * padding - k_height) / stride) + 1
    out_w = int((W + 2 * padding - k_width) / stride) + 1
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    #print(out_h,out_w)
    level1 = np.repeat(np.arange(k_height),k_width)
    level1 = np.tile(level1, C)
    everyLevels = 1 * np.repeat(np.arange(out_h),out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    
    slide1 = np.tile(np.arange(k_height), k_width)    
    slide1 = np.tile(slide1, C)
    everySlides = 1 * np.tile(np.arange(out_h),out_w)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(C), k_height*k_width).reshape(-1, 1)
    x_padded_new = np.zeros((N, C, H_padded, W_padded), dtype=grad_X_col.dtype)
    cols_reshaped = grad_X_col.reshape(C * k_height * k_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded_new, (slice(None), d, i, j), cols_reshaped)
    if padding == 0:
        return x_padded_new
    return x_padded_new[:, :, padding:-padding, padding:-padding]



class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Unlike Problem 1 MLP, here we no longer accumulate the gradient values,
        we assign new gradients directly. This means we should call update()
        every time we do forward and backward, which is fine. Consequently, in
        Problem 2 zerograd() is not needed any more.
        Compute and save the gradients wrt the parameters for update()
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.h=[]
        pass
    
    def forward(self, x, train=True):
        """
        returns ReLU(x)
        """
        if np.isscalar(x):
            self.h = np.max((x, 0))
        else:
            compare_x = np.stack((x , np.zeros(x.shape)), axis = -1)
            self.h = np.max(compare_x, axis = -1)
        return self.h
        
    def backward(self, dLoss_dout):
        """
        dLoss_dout is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        grad_out=np.multiply(dLoss_dout,self.h>0)
        return grad_out

      


class Flatten(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.input_x=[]
        pass
    def forward(self, x):
        """
        returns Flatten(x)
        """
        self.input_x_shape = x.shape #Need it in backpropagation.
        flatten_x = x.reshape(self.input_x_shape[0], -1) #(1x400)
        return flatten_x

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        return dloss.reshape(self.input_x_shape)


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.C, self.H, self.Width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6) / np.sqrt((self.C + self.num_filters) * self.k_height * self.k_width)
        self.W = np.random.uniform(-b, b, (self.num_filters, self.C, self.k_height, self.k_width))
        self.b = np.zeros((self.num_filters, 1))
        self.X=None
        self.X_col=None
        self.w_col=None
        self.grad_b=np.zeros(self.b.shape)
        self.grad_W=np.zeros(self.W.shape)
        self.grad_X=np.zeros((self.C, self.H, self.Width))
        self.s=None
        self.p=None
    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here
        """
        self.X=inputs
        m = inputs.shape[0]
        self.s=stride
        self.p=pad
        print()
        C_new = self.num_filters
        H_new = int((self.H + 2 * pad - self.k_height)/ stride) + 1
        W_new = int((self.Width + 2 * pad - self.k_width)/ stride) + 1
        
        self.X_col = im2col(inputs, self.k_height, self.k_width,pad,stride)
        #kk=X_col.shape[0]
        self.w_col = self.W.reshape((self.num_filters,-1))
        #b_col = self.b.reshape(-1, 1)
        # Perform matrix multiplication.
        #print("Filter",self.num_filters,"filter size",self.num_filters, self.k_height, self.k_width)
        #print("This is the shape of",w_col.shape)
        #print(X_col.shape)
        #print(self.b.shape)
        out =self.w_col@self.X_col+ self.b.T
        out=out.reshape(C_new,H_new,W_new,m)
        #print(np.array(np.hsplit(out, m)).shape)
        # Reshape back matrix to image.
        #out = np.array(np.2(out, m))
        #print("This shape of out",out.shape)
       
        out=np.transpose(out,(3,0,1,2))
        #out=np.array(np.hsplit(out, m)).reshape((m, C_new, H_new, W_new))
        #out=out.transpose(1,0,2,3)
        #out=np.ascontiguousarray(out)
        
        return out
        

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
      
        m=self.X.shape[0]
        # Compute bias gradient.
        self.grad_b = np.sum(dloss, axis=(0,2,3)).reshape(-1,1)
        
        # Reshape dout properly.
        #print("dloss shape",dloss.transpose(1,2,3,0).shape)
    
        #dloss = dloss.reshape( 1, dloss.shape[0]*dloss.shape[1]* dloss.shape[2] * dloss.shape[3])
        dloss = dloss.transpose(1,2,3,0).reshape(self.num_filters,-1)
        #dloss = np.concatenate(dloss, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        #print("This is w_col", self.w_col.shape,"This is loss shape",dloss.shape)
        dX_col = self.w_col.T@dloss
        self.grad_X = im2col_bw(dX_col, self.X.shape, self.k_height, self.k_width, self.p,self.s)
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        #print("This is X_col", self.X_col.shape,"This is loss shape",dloss.T.shape)
        dw_col = self.X_col@dloss.T
        # Reshape back to image (col2im).
       
        # Reshape dw_col into dw.
        self.grad_W = dw_col.reshape(1,self.C, self.k_height, self.k_width)
        
        
        #dloss=dloss.transpose(1,2,3,0).reshape(self.num_filters,-1)
        #self.grad_b=

        return [self.grad_W,self.grad_b,self.grad_X]
        
    ''' 
        N,C_out,H_out,W_out=dloss.shape()
        dloss=self.grad_b.reshape(C_out,H_out*W_out*N)
        grad_x=self.W@dloss.T
        #grad_x=grad_wrt_out@self.W.T
        self.grad_W=self.linInp.T@dloss
        #self.grad_b=np.sum(grad_wrt_out,axis=1,keepdims=True).T
        self.grad_b=np.sum(dloss,axis=0).reshape((-1,1))
        #grad_back=np.array([grad_x,grad_W,grad_b])
        #print("This is gradient",self.grad_W[1,:10])
        return [self.grad_W,self.grad_b,grad_x.T]
      ''' 
        
    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as Problem1
        Here we divide gradients by batch_size (because we will be using sum Loss
        instead of mean Loss in Problem 2 during backpropogation). Do not divide
        gradients by batch_size in step() in Problem 1.
        """
        pass

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_shape=filter_shape
        self.stride=stride
        pass

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        """
        N, C, H, W= inputs.shape
        k_height,k_width=self.filter_shape
        print()
        C_new = C
        H_new = int((H  - k_height)/ self.stride) + 1
        W_new = int((W - k_width)/ self.stride) + 1
        
        X_col = im2col(inputs, k_height, k_width,0,self.stride)
        X_col = X_col.reshape(C, X_col.shape[0]//C, -1)
        out = np.max(X_col, axis=1)

        
        #out=out.reshape(C_new,H_new,W_new,N)
        #out=np.transpose(out,(3,0,1,2))
        # Reshape back matrix to image.
        #out = np.array(np.2(out, m))
        #out=np.array(np.hsplit(out, N)).reshape((N, C_new, H_new, W_new))
        #out=out.transpose(1,0,2,3)
        #out=np.ascontiguousarray(out)
        #print("This is what we got",out)
        return out

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        pass

class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        b = np.sqrt(6) / np.sqrt(indim + outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))
        self.linInp=[]
        self.grad_W=[]
        self.grad_b=[]
        self.v_grad_W=np.zeros(self.W.shape)
        self.v_grad_b=np.zeros(self.b.shape)

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.linInp=inputs
        #print(self.W.shape)
        #print(x.shape)
        #print(self.b.shape)
        #h=np.dot(self.W.T,x.T)+self.b
        #print(self.b.shape)
        h=inputs@self.W+self.b.reshape((1,-1))
        #h=h.T
        return h
       

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        grad_x=self.W@dloss.T
        #grad_x=grad_wrt_out@self.W.T
        self.grad_W=self.linInp.T@dloss
        #self.grad_b=np.sum(grad_wrt_out,axis=1,keepdims=True).T
        self.grad_b=np.sum(dloss,axis=0).reshape((-1,1))
        #grad_back=np.array([grad_x,grad_W,grad_b])
        #print("This is gradient",self.grad_W[1,:10])
        return [self.grad_W,self.grad_b,grad_x.T]
        
        pass

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.v_grad_W=momentum_coeff*self.v_grad_W + self.grad_W/self.linInp.shape[0]
        #x=self.alpha*self.v_grad_W
        #print("This is v-gradient after",x[1,:10])
        #print("This is the shape of the momentum",self.v_grad_b.shape)
        #x=self.alpha*self.v_grad_W
        #print("This is momentum check", x[1,10])
        self.W=self.W-learning_rate*self.v_grad_W
        self.v_grad_b=momentum_coeff*self.v_grad_b + self.grad_b/self.linInp.shape[0]
        self.b=self.b-learning_rate*self.v_grad_b
        pass

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.yhat=[]
        self.p=[]
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        #x=labels.shape[0]
        self.yhat=labels
        #print(logits.shape)
        softMax=[]
        N = np.exp(logits)
        #print(N)
        D = np.sum(N,axis=1)
        D=D.reshape((-1,1))
        #print(D)
        softMax=N/D
        
        self.p=softMax
        #print(softMax)
        #Sprint(labels)
        #print(np.multiply(labels.T,softMax))
        J = np.sum(np.multiply(labels,np.log(softMax)))
      
        #for i in N:
            #SM=i/D
            #softMax.append(SM)
    
        
        return J
      

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        return np.array(self.p - self.yhat)
     
    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        pass



class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5 (or 5x5x5)
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 5x5x5
        then apply Relu
        then Conv with filter size of 5x5x5
        then apply Relu
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class MLP:
    """
    Implement as you wish, not autograded
    """
    def __init__(self):
        pass

    def forward(self, inputs, y_labels):
        pass

    def backward(self):
        pass

    def update(self,learning_rate,momentum_coeff):
        pass


# Implement the training as you wish. This part will not be autograded.
if __name__ == '__main__':
    # This part may be helpful to write the training loop
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    
    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--momentum', type=float, default = 0.9)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--conv_layers', type=int, default = 1)
    parser.add_argument('--filters', type=int, default = 1)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    train_data = trainX.reshape(-1,3,32,32).astype(np.float32)/255.0
    train_label = np.array([[i==lab for i in range(10)] for lab in trainy], np.int32)
    test_data = testX.reshape(-1,3,32,32).astype(np.float32)/255.0
    test_label = np.array([[i==lab for i in range(10)] for lab in testy], np.int32)

    num_train = len(train_data)
    num_test = len(test_data)
    batch_size = args.batch_size
    train_iter = num_train//batch_size + 1
    test_iter = num_test//batch_size + 1

    if args.conv_layers == 1:
        cnn = ConvNet()
    elif args.conv_layers == 2:
        cnn = ConvNetTwo()
    