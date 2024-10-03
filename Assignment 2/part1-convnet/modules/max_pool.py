"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # Initializing height and width of output
        H_out = int((x.shape[2] - self.kernel_size) / self.stride) + 1
        W_out = int((x.shape[3] - self.kernel_size) / self.stride) + 1
        
        out = 0 * np.random.randn(x.shape[0],x.shape[1],H_out,W_out)

        for p in range(0,x.shape[0]):
            for q in range(0,x.shape[1]):
                h = -1
                for r in range(0,x.shape[2],self.stride):
                    h += 1
                    w = -1
                    for s in range(0,x.shape[3], self.stride):
                        w+=1
                        out[p][q][h,w] = np.max(x[p][q][r:r+self.stride, s:s+self.stride]) 

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        self.dx = x * 0

        for p in range(0,x.shape[0]):
            for q in range(0,x.shape[1]):
                h=-1
                for r in range(0,x.shape[2],self.stride):
                    h+=1
                    w=-1
                    for s in range(0,x.shape[3],self.stride):
                        w+=1

                        arg_max = np.argmax(x[p][q][r:r+self.stride,s:s+self.stride])                        
                        # print('arg_max', arg_max)
                        index = np.unravel_index(arg_max,x[p][q][r:r+self.stride,s:s+self.stride].shape)
                        self.dx[p][q][r+index[0],s+index[1]] = dout[p][q][h,w]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
