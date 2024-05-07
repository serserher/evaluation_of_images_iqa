import torch
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class PerformancePredictor(Module):
    def __init__(self, BaseModel, dropout_rate=0.25):
        super(PerformancePredictor, self).__init__()
        # initialize the base model and the number of classes
        self.BaseModel = BaseModel
        
        # Determine the number of input features for the custom classifier
        if isinstance(self.BaseModel, type(mobilenet_v2())):
            num_ftrs = self.BaseModel.classifier[1].out_features
        else:
            input_size = self.BaseModel.fc.in_features
            num_ftrs = input_size

        # Define the sizes for the linear layers
        sizes = [num_ftrs, 512, 256, 128, 64, 2]
        # build the classifier head to predict the class labels
        print(sizes)
        self.criteria1 = Sequential(
            Linear(sizes[0], sizes[1]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[1], sizes[2]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[2], sizes[3]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[3], sizes[4]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[4], sizes[5])
        )
        self.criteria2 = Sequential(
            Linear(sizes[0], sizes[1]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[1], sizes[2]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[2], sizes[3]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[3], sizes[4]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[4], sizes[5])
        )
        self.criteria4 = Sequential(
            Linear(sizes[0], sizes[1]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[1], sizes[2]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[2], sizes[3]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[3], sizes[4]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[4], sizes[5])
        )
        self.criteria5 = Sequential(
            Linear(sizes[0], sizes[1]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[1], sizes[2]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[2], sizes[3]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[3], sizes[4]),
            ReLU(),
            Dropout(dropout_rate),
            Linear(sizes[4], sizes[5])
        )
        
        # This next step essentially removes the last layer (it does not remove it but it multiplies all the previous outputs by 1), keeping the convolutional
        # features of the base model without the classifying part.
        self.BaseModel.fc = Identity()
		
    def forward(self, x):
        # pass the inputs through the base model and then obtain predictions from different branches of the network
        features = self.BaseModel(x)
        perf1 = F.softmax(self.criteria1(features), dim=1)
        perf2 = F.softmax(self.criteria2(features), dim=1)
        #perf3 = F.softmax(self.criteria3(features), dim=1)
        perf4 = F.softmax(self.criteria4(features), dim=1)
        perf5 = F.softmax(self.criteria5(features), dim=1)
        performance = [perf1, perf2, perf4, perf5]
        return performance
