import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, image_size=28, kernel_size=5, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size)
        self.batchnorm2 = nn.BatchNorm2d(64)

        def conv_output_size(size, kernel_size, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1

        size_after_conv1 = conv_output_size(image_size, kernel_size)
        size_after_pool1 = size_after_conv1 // 2  # maxpool halves spatial size
        
        size_after_conv2 = conv_output_size(size_after_pool1, kernel_size)
        size_after_pool2 = size_after_conv2 // 2

        fc_input_size = 64 * size_after_pool2 * size_after_pool2

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout1 = nn.Dropout(p=0.45)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(p=0.35)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
