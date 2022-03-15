import math
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

from rbm import *


class DBN:
    def __init__(self, n_visible, layers, k, lr, max_epochs):
        """ 
        The Deep Belief Network (DBN) class
        Args:
            n_visible: Dimension of visible features layer
            layers: a list, the dimension of each hidden layer, e.g,, [500, 784]
            k: gibbs sampling steps
            lr: learning rate, remains constant through train
            max_epochs: Number of train epochs
        """
        # Instantiate DBN class constants
        #---------------------------------------------
        self.n_visible = n_visible
        self.layers = layers
        self.k = k
        self.lr = lr
        self.max_epochs = max_epochs

        # Instantiate RBM components through the layers
        #----------------------------------------------
        self.rbms = []
        rbm =  RBM(self.n_visible, self.layers[0],  self.k,self.lr, self.max_epochs) # Instantiate the first RBM
        self.rbms.append(rbm)
         
        for i in range(1, len(self.layers)):
            # Instantiate the RBM layers
            #print(i)
            rbm =  RBM(self.layers[i-1], self.layers[i],  self.k,self.lr, self.max_epochs) # Instantiate the first RBM
            self.rbms.append(rbm)
            #print("complete")
 
    def fit(self, X, valid_X):

        """ The training process of a DBN, basically we train RBMs one by one
        Args:
            X: the train images, numpy matrix
            valid_X: the valid images, numpy matrix
        """

        # zero lists for reconstruction errors
        self.te_list = np.zeros((len(self.rbms), self.max_epochs))
        self.ve_list = np.zeros((len(self.rbms), self.max_epochs))

        # iterate over all RBMs
        for i in range(len(self.rbms)):
            if i > 0:  # get new data
                train = []
                valid = []
                #print("Entering this if")
                # iterate over the RBM's h_v and sample_h method
                # to generate the train and valid data.
                for x in X:
                    h_prob_x=self.rbms[i-1].h_v(x)
                    h_sample_x=self.rbms[i-1].sample_h(h_prob_x)
                    #h_sample_x=h_sample_x.astype(int)
                    
                    train.append(h_sample_x)
                for val in valid_X:
                    h_prob_val=self.rbms[i-1].h_v(val)
                    h_sample_val=self.rbms[i-1].sample_h(h_prob_val)
                    #h_sample_val=h_sample_val.astype(int)
                    valid.append(h_sample_val)
                train=np.array(train)
                valid=np.array(valid)
                #print("complete for new data")
                #print(train)
            else:
                train = X
                #print("For OG",train)
                valid = valid_X
                

            # iterate over all epochs
            for epoch in range(self.max_epochs):
                
                shuff = shuffle_corpus(train)

                for x in shuff:
                    # update the RBM weights
                    self.rbms[i].update(x)
                    #print("complete for update")

                te = self.rbms[i].evaluate(train)
                ve = self.rbms[i].evaluate(valid)
                self.te_list[i][epoch] = te
                self.ve_list[i][epoch] = ve

                # Print optimization trajectory
                train_error = "{:0.4f}".format(te)
                valid_error = "{:0.4f}".format(ve)
                print(f"Epoch {epoch + 1} :: RBM {i + 1} :: \t " +
                      f"Train Error {train_error} :: Valid Error {valid_error}")
        return self.te_list,self.ve_list
    def infer(self,X,k=1):
        #h0_1, v0, h_sample_1, v_sample, h_prob_1, v_prob=self.rbms[0].gibbs_k(X, k=1)
        #h1_prob_1=self.rbms[0].h_v(X)
        #h1_sample_1=self.rbms[0].sample_h(h1_prob_1)
        ##Get gib sampled h1 from h2 
        
        h1_sample_1=np.random.binomial(n=1, p=0.1,size=(self.layers[0], ))
        h0_2, h0_1, h_sample_2, h_sample_1, h_prob_2, h_prob_1=self.rbms[1].gibbs_k(h1_sample_1, k)
        v_final_prob=self.rbms[0].v_h(h_sample_1)
        v_final_sample=self.rbms[0].sample_v(v_final_prob)
        return v_final_prob,v_final_sample
    

def fit_mnist_dbn(n_visible, layers, k, max_epochs, lr):
    train_data = np.genfromtxt('C:\\Users\\Anamika Shekhar\\Desktop\\Spring22\\10707\\HW2\\S22_HW2_handout_v2\\Programming\\data\\digitstrain.txt', delimiter=",")
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]
    train_X = train_X[-900:]

    valid_data = np.genfromtxt('C:\\Users\\Anamika Shekhar\\Desktop\\Spring22\\10707\\HW2\\S22_HW2_handout_v2\\Programming\\data\\digitsvalid.txt', delimiter=",")
    valid_X = valid_data[:, :-1][-300:]
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt("C:\\Users\\Anamika Shekhar\\Desktop\\Spring22\\10707\\HW2\\S22_HW2_handout_v2\\Programming\\data\\digitstest.txt", delimiter=",")
    test_X = test_data[:, :-1][-300:]
    test_Y = test_data[:, -1]

    train_X = binary_data(train_X)
    valid_X = binary_data(valid_X)
    test_X = binary_data(test_X)

    n_visible = train_X.shape[1]
    
    dbn = DBN(n_visible=n_visible, layers=layers, 
              k=k, max_epochs=max_epochs, lr=lr)
    dbn.fit(X=train_X, valid_X=valid_X)


if __name__ == "__main__":
    
    np.seterr(all='raise')
    plt.close('all')

    fit_mnist_dbn(n_visible=784, layers=[500, 784], k=5, max_epochs=10, lr=0.01)

