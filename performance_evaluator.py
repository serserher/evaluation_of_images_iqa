import torch
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid


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
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria2 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria3 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria4 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria5 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria6 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		self.criteria7 = Sequential(
			Linear(self.BaseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, 1), #binary classification per each criteria
		)
		# This next step essentially removes the last layer (it does not remove it but it multiplies all the previous outputs by 1), keeping the convolutional
		# features of the base model without the classifying part.
		self.BaseModel.fc = Identity()
		
	def forward(self, x):
		# pass the inputs through the base model and then obtain predictions from two different branches of the network
			features = self.BaseModel(x)
			perf1 = self.criteria1(features)
			perf2 = self.criteria2(features)
			perf3 = self.criteria3(features)
			perf4 = self.criteria4(features)
			perf5 = self.criteria5(features)
			perf6 = self.criteria6(features)
			perf7 = self.criteria7(features)
			performance = [perf1, perf2, perf3, perf4, perf5, perf6, perf7]
			return (performance)