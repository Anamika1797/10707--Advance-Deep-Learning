#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math
import numpy as np
import copy

# When you submit the code for autograder, comment the load cifar 10 dataset command.
# This is only for experiment.
from load_cifar import trainX, trainy, testX, testy


def random_normal_weight_init(indim, outdim):
    return np.random.normal(0,1,(indim, outdim))

def random_weight_init(indim,outdim):
    b = np.sqrt(6)/np.sqrt(indim+outdim)
    return np.random.uniform(-b,b,(indim, outdim))

def zeros_bias_init(outdim):
    return np.zeros((outdim,1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels],dtype=np.float32)


# In[2]:


class Transform:##Code this not
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
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


# In[5]:


class ReLU(Transform):##Code
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.h=[]
        pass

    def forward(self, x, train=True):
        if np.isscalar(x):
            self.h = np.max((x, 0))
        else:
            compare_x = np.stack((x , np.zeros(x.shape)), axis = -1)
            self.h = np.max(compare_x, axis = -1)
        return self.h

    def backward(self, grad_wrt_out):
        h_d=[]
        #print(self.h.shape)
        for i in self.h:
            jj=[]
            for j in i:
                if j>0:
                    jj.append(1)
                else:
                    jj.append(0)
            h_d.append(jj)
        grad_relu=np.multiply(grad_wrt_out,np.array(h_d))
            
        return grad_relu


# In[ ]:


class LinearMap(Transform):##Code
    """
    Implement this class
    feel free to use random_xxx_init() functions given on top
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.indim=indim
        self.outdim=outdim
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)
        self.b = zeros_bias_init(outdim)
        "New parameters"
        self.linInp=[]
        self.grad_W=[]
        self.grad_b=[]
        self.v_grad_W=np.zeros(self.W.shape)
        self.v_grad_b=np.zeros(self.b.shape)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        self.linInp=x
        #print(self.W.shape)
        #print(x.shape)
        #print(self.b.shape)
        #h=np.dot(self.W.T,x.T)+self.b
        #print(self.b.shape)
        h=x@self.W+self.b.reshape((1,-1))
        #h=h.T
        return h

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        #grad_x=np.dot(self.W,grad_wrt_out.T).T
        
        #print("This is backward grad",self.grad_W[0])
        #self.grad_W=np.dot(grad_wrt_out.T,self.linInp).T
        #print(self.linInp.shape)
        #print(grad_wrt_out.shape)
        grad_x=self.W@grad_wrt_out.T
        #grad_x=grad_wrt_out@self.W.T
        self.grad_W=self.linInp.T@grad_wrt_out
        #self.grad_b=np.sum(grad_wrt_out,axis=1,keepdims=True).T
        self.grad_b=np.sum(grad_wrt_out,axis=0).reshape((-1,1))
        #grad_back=np.array([grad_x,grad_W,grad_b])
        #print("This is gradient",self.grad_W[1,:10])
        return grad_x.T

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        #print(self.v_grad_W.shape)
        #alpha_W=np.full(self.grad_W.shape,self.alpha)
        #alpha_b=np.full(self.grad_b.shape,self.alpha)
        #print(self.alpha.shape)
        #print("This is the shape of W",self.grad_W.shape)
        
        #print("This is the shape of b",self.grad_b.shape)
        #x=self.alpha*self.v_grad_W
        #print("This is v-gradient before",x[1,:10])
        self.v_grad_W=self.alpha*self.v_grad_W + self.grad_W
        #x=self.alpha*self.v_grad_W
        #print("This is v-gradient after",x[1,:10])
        #print("This is the shape of the momentum",self.v_grad_b.shape)
        #x=self.alpha*self.v_grad_W
        #print("This is momentum check", x[1,10])
        self.W=self.W-self.lr*self.v_grad_W
        self.v_grad_b=self.alpha*self.v_grad_b + self.grad_b
        self.b=self.b-self.lr*self.v_grad_b
        '''
        self.v_grad_W = self.grad_W + self.v_grad_W*self.alpha
        x=self.v_grad_W*self.alpha 
        print("This is momentum check", x[1,:10])
        self.v_grad_b = self.grad_b + self.v_grad_b*self.alpha
        self.W -= self.lr*self.v_grad_W
        self.b -= self.lr*self.v_grad_b
        '''
        pass

    def zerograd(self):
        #print("This is zerograd hello")
        self.grad_W=np.zeros(self.grad_W.shape)
        #print(self.grad_W[0])np.zeros(self.grad_b.shape)
        self.grad_b=np.zeros(self.grad_b.shape)
        pass 

    def getW(self):
    # return weights
        return self.W

    def getb(self):
    # return bias
        return self.b

    def loadparams(self, w, b):
    # Used for Autograder. Do not change.
        self.W, self.b = w, b


# In[ ]:


class SoftmaxCrossEntropyLoss:##Code
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.yhat=[]
        self.p=[]
        
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
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
        J = (-1/labels.shape[0])*np.sum(np.multiply(labels,np.log(softMax)))
      
        #for i in N:
            #SM=i/D
            #softMax.append(SM)
    
        
        return J

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return (1/self.yhat.shape[0])*np.array(self.p - self.yhat)
        
       

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        
        predict=(self.p==self.p.max(axis=1).reshape(-1,1))
        accuracy=np.multiply(predict,self.yhat).sum()/self.yhat.shape[0]
        return accuracy
    


# In[ ]:


class SingleLayerMLP(Transform):##code
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.LinearMap1=LinearMap(inp,hiddenlayer,alpha,lr)
        self.RELU1=ReLU()
        self.LinearMap2=LinearMap(hiddenlayer,outp,alpha,lr)

    def forward(self, x, train=True):
        #print(x.shape)
        a1=self.LinearMap1.forward(x)
        #print(a1.shape)
        z1=self.RELU1.forward(a1)
        #print(z1.shape)
        o=self.LinearMap2.forward(z1)
        return o

    def backward(self, grad_wrt_out):
        grad_linear2_x=self.LinearMap2.backward(grad_wrt_out)
        grad_relu=self.RELU1.backward(grad_linear2_x) 
        grad_linear1_x=self.LinearMap1.backward(grad_relu)
        return grad_linear1_x

    def step(self):
        #print("Step was called")
        #print("This is before",self.LinearMap2.getW()[1,:10])
        self.LinearMap2.step()
        #print("This is after",self.LinearMap2.getW()[1,:10])
        self.LinearMap1.step()
        pass

    def zerograd(self):
        #print(self.LinearMap2.grad_W[1,:10])
        self.LinearMap2.zerograd()
        #print(self.LinearMap2.grad_W[1,:10])
        self.LinearMap1.zerograd()
        pass

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        LinearMap1_W=Ws[0]
        LinearMap1_b=bs[0]
        self.LinearMap1.loadparams(LinearMap1_W, LinearMap1_b)
        LinearMap2_W=Ws[1]
        LinearMap2_b=bs[1]
        self.LinearMap2.loadparams(LinearMap2_W, LinearMap2_b)
        
        pass

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this. 
        Return weights for first layer then second and so on...
        """
        Ws=[self.LinearMap1.getW(),self.LinearMap2.getW()]
        
        return Ws

    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this. 
        Return bias for first layer then second and so on...
        """
        bs=[self.LinearMap1.getb(),self.LinearMap2.getb()]
        return bs

class TwoLayerMLP(Transform):##code
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, inp, outp, hiddenlayers=[100,100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.LinearMap1=LinearMap(inp,hiddenlayers[0],alpha,lr)
        self.RELU1=ReLU()
        self.LinearMap2=LinearMap(hiddenlayers[0],hiddenlayers[1],alpha,lr)
        self.RELU2=ReLU()
        self.LinearMap3=LinearMap(hiddenlayers[1],outp,alpha,lr)
        pass

    def forward(self, x, train=True):
        #print(x.shape)
        a1=self.LinearMap1.forward(x)
        #print(a1.shape)
        z1=self.RELU1.forward(a1)
        #print(z1.shape)
        a2=self.LinearMap2.forward(z1)
        z2=self.RELU2.forward(a2)
        #print(z1.shape)
        a3=self.LinearMap3.forward(z2)
        return a3

    def backward(self, grad_wrt_out):
        grad_linear3_x=self.LinearMap3.backward(grad_wrt_out)
        grad_relu2=self.RELU2.backward(grad_linear3_x) 
        grad_linear2_x=self.LinearMap2.backward(grad_relu2)
        grad_relu1=self.RELU1.backward(grad_linear2_x) 
        grad_linear1_x=self.LinearMap1.backward(grad_relu1)
        return grad_linear1_x

    def step(self):
        self.LinearMap3.step()
        #print("This is after",self.LinearMap2.getW()[1,:10])
        self.LinearMap2.step()
        self.LinearMap1.step()
        pass

    def zerograd(self):
        self.LinearMap3.zerograd()
        self.LinearMap2.zerograd()
        #print(self.LinearMap2.grad_W[1,:10])
        self.LinearMap1.zerograd()
        pass

    def loadparams(self, Ws, bs):
        LinearMap1_W=Ws[0]
        LinearMap1_b=bs[0]
        self.LinearMap1.loadparams(LinearMap1_W, LinearMap1_b)
        LinearMap2_W=Ws[1]
        LinearMap2_b=bs[1]
        self.LinearMap2.loadparams(LinearMap2_W, LinearMap2_b)
        LinearMap3_W=Ws[2]
        LinearMap3_b=bs[2]
        self.LinearMap3.loadparams(LinearMap3_W, LinearMap3_b)
        pass

    def getWs(self):
        Ws=[self.LinearMap1.getW(),self.LinearMap2.getW(),self.LinearMap3.getW()]
        return Ws

    def getbs(self):
        bs=[self.LinearMap1.getb(),self.LinearMap2.getb(),self.LinearMap3.getb()]
        return bs
        



class Dropout(Transform):##Code
    """
    Implement this class
    """
    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p=p
        self.mask=None
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial
        Scale your output accordingly
        During test time, you should not apply any mask or scaling.
        """
        if train: 
            self.mask = np.random.binomial(1,self.p,x.shape)
            return x*self.mask
        return x*self.p

    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        return grad_wrt_out*self.mask



class BatchNorm(Transform):##Code
    """
    Implement this class
    """
    def __init__(self, indim, alpha=0.9, lr=0.01, mm=0.01):
        Transform.__init__(self)
        """
        You shouldn't need to edit anything in init
        """
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta
       
        """
        The following attributes will be tested
        """
        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))

        """
        gradient parameters
        """
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        """
        momentum parameters
        """
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)
        """
        inference parameters
        """
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        #print(self.out)
        #print(x)
        self.x=x
        
        
        #print(self.mean)
        #print(np.sqrt(self.var+self.eps))
        
      
        if train:
            
            self.mean=np.mean(x,axis=0).reshape(1,-1)
            self.var=np.var(x,axis=0).reshape(1,-1)
            dd=np.sqrt(self.var+self.eps)           
            self.norm=(x-self.mean)/dd            
            self.running_mean=self.alpha*self.running_mean+(1-self.alpha)*self.mean
            self.running_var=self.alpha*self.running_var+(1-self.alpha)*self.var            
            self.out=self.gamma*self.norm + self.beta
        else:
            self.out=(x-self.running_mean)/np.sqrt(self.running_var+self.eps)
        return self.out


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        N,D=self.x.shape
        X_mu=self.x-self.mean
        std_inv=1/np.sqrt(self.var+self.eps)
        #X_norm=X_mu*std_inv
        dX_norm=grad_wrt_out*self.gamma
        
        dvar=np.sum(dX_norm*X_mu,axis=0)*-.5*std_inv**3
        dmu=np.sum(dX_norm*-std_inv,axis=0)+dvar*np.mean(-2*X_mu,axis=0)
        dX=(dX_norm*std_inv)+(dvar*2*X_mu/N)+(dmu/N)
        self.dgamma=np.sum(grad_wrt_out*self.norm,axis=0)
        self.dbeta=np.sum(grad_wrt_out,axis=0)
        #print("Please work !!!")
        return dX
    
    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        Make sure your gradient step takes into account momentum.
        Use mm as the momentum parameter.
        """
        self.mgamma=self.mm*self.mgamma + self.dgamma
        #x=self.alpha*self.v_grad_W
        #print("This is v-gradient after",x[1,:10])
        #print("This is the shape of the momentum",self.v_grad_b.shape)
        #x=self.alpha*self.v_grad_W
        #print("This is momentum check", x[1,10])
        self.gamma=self.gamma-self.lr*self.mgamma
        self.mbeta=self.mm*self.mbeta + self.dbeta
        self.beta=self.beta-self.lr*self.mbeta
        pass

    def zerograd(self):
        # reset parameters
        self.dgamma=np.zeros(self.dgamma.shape)
        #print(self.grad_W[0])np.zeros(self.grad_b.shape)
        self.dbeta=np.zeros(self.dbeta.shape)
        pass

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta


if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    from load_cifar import trainX, trainy, testX, testy

    trainX = trainX.astype(float)/255.0 #has shape (batch_size x indim)
    testX = testX.astype(float)/255.0  #has shape (batch_size x indim)
    
    trainLabels = labels2onehot(trainy)
    testLabels = labels2onehot(testy)
    #print(trainy.shape)
    '''
    New Variables
    '''
    epochs=50
    bathc_size=128
    learning_rate=0.001
    momentum=0.9
    def random_mini_batches(X, Y, mini_batch_size = 128):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        #np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        #print(m)    
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        #print("This is per",len(permutation))
        #print("this is shape of Y",Y.shape)
        shuffled_Y = Y[permutation, :]
        
        #inc = mini_batch_size
    
        # Step 2 - Partition (shuffled_X, shuffled_Y).
        # Cases with a complete mini batch size only i.e each of 64 examples.
        num_complete_minibatches = math.floor(m / mini_batch_size) 
        #print("Total Batches",num_complete_minibatches)# number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
            mini_batch_Y =shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % mini_batch_size != 0:
            y1=mini_batch_size*math.floor(m / mini_batch_size)            
            mini_batch_X =  shuffled_X[y1:,:]
            mini_batch_Y =shuffled_Y[y1:,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    '''
    SLP=SingleLayerMLP(trainX.shape[1],10,1000,momentum,learning_rate)
    loss=SoftmaxCrossEntropyLoss()
    test_cost_SLP=[]
    train_cost_SLP=[]
    test_acc_SLP=[]
    train_acc_SLP=[]
    for e in range(epochs):
        minibatches = random_mini_batches(trainX, trainLabels, 128)
        for batchs in minibatches:
            (minibatch_X, minibatch_Y) = batchs
            logits=SLP.forward(minibatch_X)
            loss_train=loss.forward(logits, minibatch_Y)
            grad_o=loss.backward()
            SLP.backward(grad_o)
            SLP.step()
            
        logits=SLP.forward(trainX)
        loss_train=loss.forward(logits, trainLabels)
        train_cost_SLP.append(loss_train)
        train_acc_SLP.append(loss.getAccu())
        logits=SLP.forward(testX,train=False)
        loss_test=loss.forward(logits, testLabels)
        test_cost_SLP.append(loss_test)
        test_acc_SLP.append(loss.getAccu())
        print("Epoch done !!")
        



#%%
    np.savetxt('SLMLP_train_loss.csv',np.array(train_cost_SLP),delimiter=",")
    np.savetxt('SLMLP_test_loss.csv',np.array(test_cost_SLP),delimiter=",")
    np.savetxt('SLMLP_train_acc.csv',np.array(train_acc_SLP),delimiter=",")
    np.savetxt('SLMLP_test_acc.csv',np.array(test_acc_SLP),delimiter=",")
#%%
    from matplotlib import pyplot as plt
    
    plt.plot(range(50),train_cost_SLP,label='Train')
    plt.plot(range(50),test_cost_SLP,label='Test')
    plt.title('Single Multi-Layer Perceptron Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
#%%    
    plt.plot(range(50),train_acc_SLP,label='Train')
    plt.plot(range(50),test_acc_SLP,label='Test')
    plt.title('Single Multi-Layer Perceptron Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    '''
    TLP=TwoLayerMLP(trainX.shape[1],10,[1000,1000],momentum,learning_rate)
    loss=SoftmaxCrossEntropyLoss()
    test_cost_TLP=[]
    train_cost_TLP  =[]
    test_acc_TLP=[]
    train_acc_TLP=[]
    for e in range(epochs):
        minibatches = random_mini_batches(trainX, trainLabels, 128)
        for batchs in minibatches:
            (minibatch_X, minibatch_Y) = batchs
            logits=TLP.forward(minibatch_X)
            loss_train=loss.forward(logits, minibatch_Y)
            grad_o=loss.backward()
            TLP.backward(grad_o)
            TLP.step()
            
        logits=TLP.forward(trainX)
        loss_train=loss.forward(logits, trainLabels)
        train_cost_TLP.append(loss_train)
        train_acc_TLP.append(loss.getAccu())
        logits=TLP.forward(testX,train=False)
        loss_test=loss.forward(logits, testLabels)
        test_cost_TLP.append(loss_test)
        test_acc_TLP.append(loss.getAccu())
        print("Epoch done !!",e)
    from matplotlib import pyplot as plt
    
    plt.plot(range(50),train_cost_TLP,label='Train')
    plt.plot(range(50),test_cost_TLP,label='Test')
    plt.title('Two Multi-Layer Perceptron Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
    
    plt.plot(range(50),train_acc_TLP,label='Train')
    plt.plot(range(50),test_acc_TLP,label='Test')
    plt.title('Two Multi-Layer Perceptron Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #%%
    np.savetxt('TLMLP_train_loss.csv',np.array(train_cost_TLP),delimiter=",")
    np.savetxt('TLMLP_test_loss.csv',np.array(test_cost_TLP),delimiter=",")
    np.savetxt('TLMLP_train_acc.csv',np.array(train_acc_TLP),delimiter=",")
    np.savetxt('TLMLP_test_acc.csv',np.array(test_acc_TLP),delimiter=",")
