""" 			  		 			     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

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

# Do not use packages that are not in standard distribution of python
import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        # This provides 128 hidden nodes
        result = np.dot(X,self.weights['W1'])
        result_w_b = result + self.weights['b1']

        # Sigmoid for layer 1
        sig_lay_1 =_baseNetwork.sigmoid(self, result_w_b)

        # This provides 64 rows x 10 of the classes score.
        result_2 = np.dot(sig_lay_1,self.weights['W2'])
        result_w_b_2 = result_2 + self.weights['b2']
        
        # Apply Softmax to get the probability; shape: 64x10
        softmaxed = _baseNetwork.softmax(self, result_w_b_2)
        
        # Apply Cross Entropy; scalar
        loss = _baseNetwork.cross_entropy_loss(self,softmaxed,y)

        # Apply Accuracy; scalar
        accuracy = _baseNetwork.compute_accuracy(self,softmaxed,y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################  

        # p -> softmax/cross entropy
        # p = q + b2
        # q = W2 * r
        # r = sigmoid(s)
        # s = t + b1
        # t = x * w1

        # Required Grad Shape
        # w1 grad shape expected (784, 128)
        # w2 grad shape expected (128, 10)
        # b1 grad shape expected (128,)
        # b2 grad shape expected (10,)

        # dL/dP
        for i in range(0,len(y)):
            softmaxed[i, y[i]] = softmaxed[i, y[i]] - 1

        batch_size = len(y)
        softmaxed = softmaxed/batch_size

        # dL/db2 = dL/dP * dP/db2
        # dL/db2 = dL/dP * 1
        # softmaxed shape = 64 * 10; softmaxed_t = 10 * 64
        softmaxed_t = softmaxed.transpose()
        # dl_db2 = 10 * 64 dot 64
        dL_db2 = np.dot(softmaxed_t,np.ones(len(softmaxed)))
        self.gradients['b2'] = dL_db2

        # dL/dw2 = dL/dp * dp/dq * dq/dW2 
        # dL/dw2 = dL/dp * sigmoid(s)
        # dL/dp = 64 * 10, sigmoid(s) = 64 * 128
        dL_dW2 = np.dot(sig_lay_1.transpose(),softmaxed)
        self.gradients['W2'] = dL_dW2

        # dL/db1 = dL/dp * dp/dq * dq/dr * dr/ds * ds/db1
        # dL/db1 = dL/dp * 1 * W2 * sigmoid_dev * 1
        # dL/dp = 64 * 10, W2 = 128 * 10, sigmoid_dev = 64 * 128, b1 = 128 * 1
        shape_64_128 = np.dot(softmaxed,self.weights['W2'].transpose())
        shape_64_128_multiply = np.multiply(shape_64_128, self.sigmoid_dev(result_w_b))       
        shape_128_64 = shape_64_128_multiply.transpose()
        dL_db1 = np.dot(shape_128_64,np.ones(len(X)))
        self.gradients['b1'] = dL_db1

        # dL/dw1 = dL/dp * dp/dq * dq/dr * dr/ds * ds/dt * dt/dx1
        # dL/dw1 = dL/db1 * 1 * W2 * sigmoid_dev * 1 * x
        # dL/dp = 64 * 10, W2 = 128 * 10, sigmoid_dev = 64 * 128, x = 64 * 784, W1 = 784*128
        # dL/db1 = 128 * 1
        dL_dw1 = np.dot(X.transpose(),shape_64_128_multiply)
        self.gradients['W1'] = dL_dw1

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
