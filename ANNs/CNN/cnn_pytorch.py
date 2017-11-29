import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
import torch.nn.functional as F




def _calculate_conv_size(pre_size, conv_kernel_size, stride_step = (1,1)):
    '''stride step is set to (1,1)'''
    return (pre_size[0]-conv_kernel_size+1, pre_size[1]-conv_kernel_size+1)

def _pooling(conv_size,  spatial_extent, stride):
    # TODO, the result might be float
    width = (conv_size[0]-spatial_extent)/stride+1
    height = (conv_size[1]-spatial_extent)/stride+1
    if (not width.is_integer()) or (not height.is_integer()):
        print ("width: {}, height: {}".format(width, height))
        raise Exception("please redesign the F or S of pooling")
    else:
        return (width, height)

# image_size = (100, 1, 28, 28)
# conv1_size = _calculate_conv_size(image_size, 5)
# conv1_size = _pooling(conv1_size, 2, 2)
# conv2_size = _calculate_conv_size(conv1_size, 5)
# conv2_size = _pooling(conv2_size, 2, 2)



class CNNPytorch(nn.Module):

    def __init__(self, image_size):
        super(CNNPytorch, self).__init__()

        # image-2d-size
        image_2d_size = tuple(image_size[2:4])

        # set channel
        input_img_channel_num = int(image_size[1])
        conv1_channel_num = 6
        conv2_channel_num = 16

        # set kernel size
        conv1_kernel_size = 5
        conv2_kernel_size = 5

        # set pooling stride, spatial_extent
        F = 2
        S = 2


        # calculate the flatten size for the conv output
        conv1_size = _calculate_conv_size(image_2d_size, conv1_kernel_size)
        conv1_size = _pooling(conv1_size, F, S)
        conv2_size = _calculate_conv_size(conv1_size, conv2_kernel_size)
        conv2_size = _pooling(conv2_size, F, S)
        conv_output_flatten_size = int(conv2_channel_num*conv2_size[0]*conv2_size[0])


        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_img_channel_num, conv1_channel_num, conv1_kernel_size)
        self.conv2 = nn.Conv2d(conv1_channel_num, conv2_channel_num, conv2_kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(conv_output_flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# if __name__ == '__main__':
#     image_size = (100, 1, 28, 28)
#     net = CNNPytorch(image_size)
#     print(net)