# imports
import torch.nn as nn
import torch.nn.functional as F


class Net3D(nn.Module):
    def __init__(self):
        super(Net3D, self).__init__()

        #calling conv3d module for convolution
        self.conv1 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2)

        #calling MaxPool3d module for max pooling with downsampling of 2
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2)

        self.conv2 =  nn.Conv3d(in_channels = 32, out_channels = 50, kernel_size = (1,3,3), stride = 2)

        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        #fully connected layer
        self.fc1 = nn.Linear(50*7*7, 1024)
        self.fc2 = nn.Linear(1024, 5)


    # defining the structure of the network
    def forward(self, x):
        # Applying relu activation after each conv layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Reshaping to 1d for giving input to fully connected units
        x = x.view(-1, 50*7*7)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))

        # Output layer (no activation)
        x = self.fc2(x)

        return x

class Model_Improved(nn.Module):
    def __init__(self):
        super(Model_Improved, self).__init__()

        #calling conv3d module for convolution
        self.conv1 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2)

        self.bn1 = nn.BatchNorm3d(32)

        #calling MaxPool3d module for max pooling with downsampling of 2
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2)

        self.conv2 =  nn.Conv3d(in_channels = 32, out_channels = 128, kernel_size = (1,3,3), stride = 2)

        self.bn2 = nn.BatchNorm3d(128)

        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        # fully connected layer
        self.fc1 = nn.Linear(128*7*7, 1024)

        self.fc2 = nn.Linear(1024, 5)



    # defining the structure of the network
    def forward(self, x):
        # Applying relu activation after each conv layer
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Reshaping to 1d for giving input to fully connected units
        x = x.view(-1, 128*7*7)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))

        # Output layer (no activation)
        x = self.fc2(x)

        return x

