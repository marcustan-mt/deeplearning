"""
Linear Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from linear.py!")

class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        # print('xshape',x.shape)
        # print('selfindim',self.in_dim)

        # print(self.weight)   
        # print(x.shape)
        # print(self.in_dim)
        
        flattened_array = x.reshape(x.shape[0], self.in_dim)
        # print('d')
        out = np.dot(flattened_array,self.weight)
        # print('d1')
        out = out + self.bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################

        # dout.shape = 10x5, bias.shape = 5 x 1, db = 5 x 1
        self.db = np.sum(dout ,axis = 0)
        
        # dout.shape = 10x5, dweight = 6x5, x = 10 x 6
        self.dw = np.dot(x.reshape(x.shape[0], self.in_dim).transpose(), dout)

        # dout.shape = 10x5, x.shape = 10 x 6, weight = 6x5
        self.dx = np.dot(dout,self.weight.transpose())

        if len(x.shape) == 2:
            self.dx = self.dx.reshape(self.dx.shape[0], x.shape[1])
        else:
            list_shape = []
            list_shape.append(self.dx.shape[0])
            for i in range(1,len(x.shape)):
                list_shape.append(x.shape[i])
            
            self.dx = self.dx.reshape(*list_shape)

        
        # self.dx = self.dx.reshape(self.dx.shape[0],x.shape[1],x.shape[2])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
