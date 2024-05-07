import torch
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class PerformancePredictor(Module):
    def __init__(self, BaseModel, dropout_rate=0.25):
        super(PerformancePredictor, self).__init__()
        # initialize the base model and the number of classes
        self.BaseModel = BaseModel
        """
        # Determine the number of input features for the custom classifier
        if isinstance(self.BaseModel, type(mobilenet_v2())):
            num_ftrs = self.BaseModel.classifier[1].out_features
        else:
            input_size = self.BaseModel.fc.in_features
            num_ftrs = input_size
        """
        # Define the sizes for the convolutional and fully connected layers
        conv_sizes = [512, 256, 128]
        fc_sizes = [64, 2]
                
        # build the classifier head to predict the class labels
        self.criteria1_conv = Sequential(
            Conv2d(conv_sizes[0], conv_sizes[1], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(conv_sizes[1], conv_sizes[2], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        
        self.criteria2_conv = Sequential(
            Conv2d(conv_sizes[0], conv_sizes[1], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(conv_sizes[1], conv_sizes[2], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        
        self.criteria4_conv = Sequential(
            Conv2d(conv_sizes[0], conv_sizes[1], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(conv_sizes[1], conv_sizes[2], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        
        self.criteria5_conv = Sequential(
            Conv2d(conv_sizes[0], conv_sizes[1], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(conv_sizes[1], conv_sizes[2], kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        
        self.criteria1_lin = Sequential(
            Linear(128*6*6, fc_sizes[0]),
            ReLU(),
            Linear(fc_sizes[0], fc_sizes[1])
        )
        
        self.criteria2_lin = Sequential(
            Linear(128*6*6, fc_sizes[0]),
            ReLU(),
            Linear(fc_sizes[0], fc_sizes[1])
        )
        
        self.criteria4_lin = Sequential(
            Linear(128*6*6, fc_sizes[0]),
            ReLU(),
            Linear(fc_sizes[0], fc_sizes[1])
        )
        
        self.criteria5_lin = Sequential(
            Linear(128*6*6, fc_sizes[0]),
            ReLU(),
            Linear(fc_sizes[0], fc_sizes[1])
        )
        

		
    def forward(self, x):
        # pass the inputs through the base model and then obtain predictions from different branches of the network
        features = self.BaseModel(x)

        perf1_conv = self.criteria1_conv(features)
        perf1_conv = perf1_conv.view(perf1_conv.size(0), -1)
        perf1 = F.softmax(self.criteria1_lin(perf1_conv), dim=1)
        
        perf2_conv = self.criteria2_conv(features)
        perf2_conv = perf2_conv.view(perf2_conv.size(0), -1)
        perf2 = F.softmax(self.criteria2_lin(perf2_conv), dim=1)
        
        perf4_conv = self.criteria4_conv(features)
        perf4_conv = perf4_conv.view(perf4_conv.size(0), -1)
        perf4 = F.softmax(self.criteria4_lin(perf4_conv), dim=1)
        
        perf5_conv = self.criteria5_conv(features)
        perf5_conv = perf5_conv.view(perf5_conv.size(0), -1)
        perf5 = F.softmax(self.criteria5_lin(perf5_conv), dim=1)
        

        performance = [perf1, perf2, perf4, perf5]
        return performance
    