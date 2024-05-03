import torch
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
import torch.nn.functional as F


class PerformancePredictor(Module):
	def __init__(self, BaseModel):
		super(PerformancePredictor, self).__init__()
		# initialize the base model and the number of classes
		self.BaseModel = BaseModel
		# build the classifier head to predict the class labels
		self.criteria1 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria2 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria3 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria4 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria5 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria6 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		self.criteria7 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 2), #binary classification per each criteria
		)
		# This next step essentially removes the last layer (it does not remove it but it multiplies all the previous outputs by 1), keeping the convolutional
		# features of the base model without the classifying part.
		self.BaseModel.fc = Identity()
		
	def forward(self, x):
		# pass the inputs through the base model and then obtain predictions from two different branches of the network
		features = self.BaseModel(x)
		perf1 = F.softmax(self.criteria1(features), dim=1)
		perf2 = F.softmax(self.criteria2(features), dim=1)
		perf3 = F.softmax(self.criteria3(features), dim=1)
		perf4 = F.softmax(self.criteria4(features), dim=1)
		perf5 = F.softmax(self.criteria5(features), dim=1)
		perf6 = F.softmax(self.criteria6(features), dim=1)
		perf7 = F.softmax(self.criteria7(features), dim=1)
		performance = [perf1, perf2, perf3, perf4, perf5, perf6, perf7]
		return (performance)