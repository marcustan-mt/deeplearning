"""
CovNet Module.  (c) 2021 Georgia Tech

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

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from conv_classifier.py!")

class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################

        i = 0
        while i < len(self.modules):
            
            x = self.modules[i].forward(x)
            i+=1

        out_1, out_2 = self.criterion.forward(x,y)

        probs = out_1

        loss = out_2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        
        # First dout
        # self.criterion.backward().dx
        self.criterion.backward()
        self.dx = self.criterion.dx

        counter = len(self.modules) - 1
        while counter >= 0:
            self.modules[counter].backward(self.dx)
            self.dx = self.modules[counter].dx
            # self.dw = self.modules[counter].dw
            # self.db = self.modules[counter].db
            
            counter -= 1

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
