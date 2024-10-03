"""
2d Convolution Module.  (c) 2021 Georgia Tech

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
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        # Padding the X only on the height and width
        padded_val = self.padding
        pad_val = ((0,0),(0,0),(padded_val,padded_val),(padded_val,padded_val)) #(top, bottom), (left, right)
        new_x = np.pad(x, pad_val, 'constant')

        # Assigning Values For X
        num_datapoints = new_x.shape[0]
        rgb = new_x.shape[1]
        height = new_x.shape[2]
        width = new_x.shape[3]
        # print('previousheight', x.shape[2], 'currentehight', new_x.shape[2])
        
        # Assigning Values for Convolve Filters
        filters = self.weight.shape[0]
        channel = self.weight.shape[1]
        kernel_height = self.weight.shape[2]
        kernel_width = self.weight.shape[3]

        # Initializing results that we will return
        height_output = int(((height - kernel_height)/self.stride) + 1)
        width_output = int(((width - kernel_width)/self.stride) + 1)
        result = 0 * np.random.randn(num_datapoints, filters, height_output, width_output)

        # Test case #1: 4 images, 3 channels, 7x7 Pixels
            # Weight: 2 filters, 3 channels, 3x3 Kernel size

        # Test case #2: 4 images, 3 channels, 7x7 Pixels
            # Weight: 2 filters, 3 channels, 3x3 Kernel size

        # Test case #3: 2 images, 3 channels, 6x6 pixels
            # Weight: 2 filters, 3 channels, 4x4 Kernel size

        # Looping over each data point
        for p in range(0,num_datapoints):
        
            h = -1 # Initializing the height values to use as index for the final output later
            # Looping over each height point
            for r in range(0,height,self.stride):
                
                h+=1 # Adding h for every incremental change in height
                w = -1 # Initializing the weight values to use as index for the final output later
                # Looping over each width point
                for s in range(0,width,self.stride):
                    
                    w+=1 # Adding w for every incremental change in weight
                    # Looping over each layer of the convolve filters:
                    for t in range(0, filters):
                        
                        output = 0 
                        # Looping over each colour
                        for q in range(0,rgb):

                            # print('this is for test case where x.shape = ', x.shape)

                            # print('p or datapoint', p)
                            # print('q or channel', q)
                            # print('r', r, 'kernel height', r+kernel_height)
                            # print('kernel_height', kernel_height)
                            # print('kernel_width', kernel_width)
                            # print('s',s, 's+kernel_width', s+kernel_width)
                                                        
                            if (r + kernel_height) > height or (s + kernel_width) > width:
                                # print('breaking')
                                continue
                                
                            else:
                                var = new_x[p][q][r:r+kernel_height,s:s+kernel_width]
                                output += (np.sum(var*self.weight[t][q]))

                                result[p][t][h,w] = output


        # Adding Biases - initializing bs
        add_bs = self.bias

        # Looping over all the datapoints
        for p in range(0, num_datapoints):

            # Looping over all the filters
            for q in range(0, filters):

                # Looping over all the height
                for r in range(0, height_output):

                    # Looping over all the width
                    for s in range(0,width_output):

                        # Adding biases
                        result[p][q][r][s] = result[p][q][r][s] + add_bs[q]
        
        # Reassigning result as the final output
        out = result
                        

        # print('p_listh ere')
        # print(len(p_list))
        # print(len(p_list[0]))
        # print(len(p_list[0][0]))
        # print(len(p_list[0][0][0]))           


        # print('here')
        # print(self.in_channels)
        # print(self.out_channels)
        # print(self.kernel_size)
        # print(self.stride)
        # print(self.padding)

        # print('shape', x.shape[0])
        # print('here1')
        # print(len(x))
        # # print(len(x[0]))
        # print((x[3]))
        # print(len(x[1]))
        # print(len(x[2]))
        # print(len(x[3]))
        # print(len(x))
        # print(len(self.weight[0]))
        # print(type(self.weight[0][0]))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        ### Differential calculations ###
        # d(loss)/d(output) = dout
        # d(loss)/d(b) = dout
        # d(loss)/d(w) = dout * d(x.w)/dw
        # d(loss)/d(x) = dout * d(x.w)/dx

        # Padding the X only on the height and width
        padded_val = self.padding
        pad_val = ((0,0),(0,0),(padded_val,padded_val),(padded_val,padded_val))
        new_x = np.pad(x, pad_val, 'constant')

        # Padding the dout only on the height and width
        # pad_val = ((0,0),(0,0),(padded_val,padded_val),(padded_val,padded_val))
        # new_dout = np.pad(dout, pad_val, 'constant')

        # Assigning Values For X
        num_datapoints = new_x.shape[0]
        rgb = new_x.shape[1]
        height = new_x.shape[2]
        width = new_x.shape[3]

        # print('previousheight', x.shape[2], 'currentehight', new_x.shape[2])
        
        # Assigning Values for Convolve Filters
        filters = self.weight.shape[0]
        channel = self.weight.shape[1]
        kernel_height = self.weight.shape[2]
        kernel_width = self.weight.shape[3]

        # Assigning dout values
        dout_num_data = dout.shape[0]
        dout_kernel = dout.shape[1]
        dout_height = dout.shape[2]
        dout_width = dout.shape[3]

        # Flipping Kernel 180 degrees
        flipped_w = np.flip(self.weight, axis=(2, 3))

        ### Differential of Loss with respect to W ###
        # Initializing results that we will return
        result = 0 * np.random.randn(filters, rgb, kernel_height, kernel_width)

        # Looping over each filter
        for t in range(0, filters):

            # Looping over each colour
            for q in range(0,rgb):

                output = 0
                # Looping over each height point
                for r in range(0,dout_height,self.stride):
            
                    # Looping over each width point
                    for s in range(0,dout_width,self.stride):
                    
                        # Looping over each data point:
                        for p in range(0,num_datapoints):             
                            var = new_x[p][q][r:r+kernel_height,s:s+kernel_width]
                            dout_val = dout[p][t][r,s]

                            output += var * dout_val


                result[t][q] = output

        self.dw = result

        ### Differential of Loss with respect to X ###
        # Initializing results that we will return
        result = 0 * new_x

        # Looping over each data point
        for p in range(0,num_datapoints): 

            # Looping over each colour
            for q in range(0,rgb):
                
                output = 0
                for t in range(0, filters):

                    # Looping over each height point on the original X
                    for r in range(0,x.shape[2],self.stride):
                
                        # Looping over each width point on the original X
                        for s in range(0,x.shape[3],self.stride):

                            # print('ssssssssssssss',s)
                            
                            dout_val = self.weight[t][q] * dout[p][t][r,s] # Matrix multiply by scalar
                            # print('dout_val', dout_val.shape)

                            # print('flipped_w',flipped_w.shape)
                            # print('r', r)
                            # print('r+kh',r+kernel_height)
                            # print('s',s)
                            # print('s+kw',s+kernel_width)

                            # print('resultshape',result[p][q].shape)
                            # print(result[p][q][r:r+kernel_height,s:s+kernel_width])
                            # print(dout_val)
                            
                            result[p][q][r:r+kernel_height,s:s+kernel_width] = result[p][q][r:r+kernel_height,s:s+kernel_width] + dout_val


        # Removing padding
        if self.padding > 0:
            for i in range(0,self.padding):    
                result = np.delete(result, 0, axis=2)
                result = np.delete(result, -1, axis=2)
                result = np.delete(result, 0, axis=3)
                result = np.delete(result, -1, axis=3)
        else:
            result = result

        self.dx = result


        ### Differential of Loss with respect to Bias ###
        # self.db with shape (2) = dout with shape(4,2,5,5)
        db_list = []
        for i in range(0, filters):
            
            val = 0
            for j in range(0,num_datapoints):
                val = val + np.sum(dout[j][i])

            db_list.append(val)

        self.db = np.array(db_list)

        



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
