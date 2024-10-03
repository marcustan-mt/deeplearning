"""
MyModel model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        # Generic initiations
        # ReLU
        self.relu = nn.ReLU()

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding = 1)




        # Layer 1: Output = 128 x 16 x 32 x 32
        self.layer_1_conv_3 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=3, stride=1, padding=1)

        # Layer 1:Output = 128 x 16 x 32 x 32
        self.layer_1_conv_5 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=5, stride=1, padding=2)
        
        # Layer 1:Output = 128 x 16 x 32 x 32
        self.layer_1_conv_7 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=7, stride=1, padding=3)
        
        # Adding Residuals
        self.residual = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=1, stride=1, bias = False)







        # Layer 2: Output = 128 x 128 x 16 x 16
        self.layer_2_conv_3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=2, padding=1)

        # Layer 2:Output = 128 x 128 x 16 x 16
        self.layer_2_conv_5 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        
        # Layer 2:Output = 128 x 128 x 16 x 16
        self.layer_2_conv_7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=2, stride=2, padding=0)

        # Adding Residuals
        self.residual_2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=1, stride=1, bias = False)

        # Conduct pooling to go from 32x32 to 16x16
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)







        # Layer 3: Output = 128 x 128 x 8 x 8
        self.layer_3_conv_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=2, stride=2, padding=0)

        # Layer 2:Output = 128 x 128 x 8 x 8
        self.layer_3_conv_5 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, stride=2, padding=1)
        
        # Layer 2:Output = 128 x 128 x 8 x 8
        self.layer_3_conv_7 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=4, stride=2, padding=1)

        # Adding Residuals
        self.residual_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=1, stride=1, bias = False)

        # Conduct pooling to go from 32x32 to 16x16
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)



        # Linear Connected Output
        self.fc1 = nn.Linear(in_features=131072, out_features=10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        # 128 x 16 x 32 x 32
        lay_1_input = self.residual(x)
        
        # Conv Layer 1 - 3
        layer_1_3 = self.layer_1_conv_3(x)
        relu_1_3 = self.relu(layer_1_3)
        result_1_3 = relu_1_3 + lay_1_input

        # Conv Layer 1 -5
        layer_1_5 = self.layer_1_conv_5(x)
        relu_1_5 = self.relu(layer_1_5)
        result_1_5 = relu_1_5 + lay_1_input

        # Conv Layer 1 - 7
        layer_1_7 = self.layer_1_conv_7(x)
        relu_1_7 = self.relu(layer_1_7)
        result_1_7 = relu_1_7 + lay_1_input

        # Conv Layer 1 - Max Pool
        # 128 x 16 x 32 x 32
        mp = self.pool(lay_1_input)
        relu_mp = self.relu(mp)
        result_mp = relu_mp + lay_1_input

        # Concatenated Result
        x = torch.cat((result_1_3, result_1_5, result_1_7, result_mp), dim=1)   
        x = self.relu(x)







        # Output: 128 x 128 x 32 x 32
        lay_2_input = self.residual_2(x)

        # Output: 128 x 128 x 16 x 16
        lay_2_input = self.max_pool_2(lay_2_input)
        
        # Conv Layer 2 - 3
        layer_2_3 = self.layer_2_conv_3(x)
        relu_2_3 = self.relu(layer_2_3)
        result_2_3 = relu_2_3 + lay_2_input

        # Conv Layer 2 -5
        layer_2_5 = self.layer_2_conv_5(x)
        relu_2_5 = self.relu(layer_2_5)
        result_2_5 = relu_2_5 + lay_2_input


        # Conv Layer 2 - 7
        layer_2_7 = self.layer_2_conv_7(x)
        relu_2_7 = self.relu(layer_2_7)
        result_2_7 = relu_2_7 + lay_2_input

        # Conv Layer 2 - Max Pool
        mp = self.pool(lay_2_input)
        relu_mp = self.relu(mp)
        result_mp = relu_mp + lay_2_input

        # Concatenated Result
        x = torch.cat((result_2_3, result_2_5, result_2_7, result_mp), dim=1) 
        x = self.relu(x) 




        # Output = 128 x 512 x 16 x 16
        lay_3_input = self.residual_3(x)
        
        # Output: 128 x 512 x 8 x 8
        lay_3_input = self.max_pool_2(lay_3_input)
        
        # Conv Layer 3 - 3
        layer_3_3 = self.layer_3_conv_3(x)
        relu_3_3 = self.relu(layer_3_3)
        result_3_3 = relu_3_3 + lay_3_input

        # Conv Layer 3 -5
        layer_3_5 = self.layer_3_conv_5(x)
        relu_3_5 = self.relu(layer_3_5)
        result_3_5 = relu_3_5 + lay_3_input

        # Conv Layer 2 - 7
        layer_3_7 = self.layer_3_conv_7(x)
        relu_3_7 = self.relu(layer_3_7)
        result_3_7 = relu_3_7 + lay_3_input

        # Conv Layer 2 - Max Pool
        mp = self.pool(lay_3_input)
        relu_mp = self.relu(mp)
        result_mp = relu_mp + lay_3_input

        # Concatenated Result
        x = torch.cat((result_3_3, result_3_5, result_3_7, result_mp), dim=1) 
        concatenated_tensor = self.relu(x) 



        flattened = torch.flatten(concatenated_tensor, 1)
        outs = self.fc1(flattened)



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
