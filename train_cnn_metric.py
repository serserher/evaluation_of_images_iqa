from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
from performance_evaluator import PerformancePredictor
from tqdm import tqdm
from torchvision.models import resnet50, resnet18
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

#PARAMETER INITIALIZATION
batch_size = 4
NUM_EPOCHS = 30
INIT_LR = 1e-4

loss_function = CrossEntropyLoss()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=MEAN, std=STD)
])
dataset = ImageDataset ('/home/sergio/Thesis_Sergio/inference/output/inference', 'labels.csv', transforms=transforms)
train_loader = DataLoader(dataset, batch_size = batch_size)

#DECLARE THE MODEL
base_model = resnet18(pretrained = True)
for param in base_model.parameters():
	param.requires_grad = False

perf_evaluator_model = PerformancePredictor(base_model)
perf_evaluator_model = perf_evaluator_model.to(DEVICE)

#DECLARE THE OPTIMIZER
optimizer = Adam(perf_evaluator_model.parameters(), lr=INIT_LR, weight_decay=1e-5) 

    #initialize the loss for each criteria
total_loss_criteria1 = []
total_loss_criteria2 = []
total_loss_criteria3 = []
total_loss_criteria4 = []
total_loss_criteria5 = []
total_loss_criteria6 = []
total_loss_criteria7 = []
    
for e in tqdm(range(NUM_EPOCHS)):
    perf_evaluator_model.train()
    
    #initialize the number of correct evaluations during the training
    train_correct_criteria1 = 0 #'Relative position and orientation between neighboring buildings'
    train_correct_criteria2 = 0 #'Position and orientation of buildings in relation to closest road/s'
    train_correct_criteria3 = 0 #'Building types in relation to underlying terrain type'
    train_correct_criteria4 = 0 #'Integrity of edges'
    train_correct_criteria5 = 0 #'Straightness of edges'
    train_correct_criteria6 = 0 #'Size relative to type'
    train_correct_criteria7 = 0 #'Conservation of color codin'
    
    for (images, criteria) in train_loader:
        images = images.to(DEVICE)
        predictions = perf_evaluator_model(images)
        predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions] 
        
        #we now calculate the losses

        loss_criteria1 = loss_function(predictions_squeezed[0], criteria['Relative position and orientation between neighboring buildings'].to(DEVICE))
        train_correct_criteria1 += ((predictions_squeezed[0] >= 0.5) == criteria['Relative position and orientation between neighboring buildings'].to(DEVICE)).type(torch.float).sum().item()
        
        loss_criteria2 = loss_function(predictions_squeezed[1], criteria['Position and orientation of buildings in relation to closest road/s'].to(DEVICE))
        train_correct_criteria2 += ((predictions_squeezed[1] >= 0.5) == criteria['Position and orientation of buildings in relation to closest road/s'].to(DEVICE)).type(torch.float).sum().item()
        
        loss_criteria3 = loss_function(predictions_squeezed[2], criteria['Building types in relation to underlying terrain type'].to(DEVICE))
        train_correct_criteria3 += ((predictions_squeezed[2] >= 0.5) == criteria['Building types in relation to underlying terrain type'].to(DEVICE)).type(torch.float).sum().item()

        loss_criteria4 = loss_function(predictions_squeezed[3], criteria['Integrity of edges'].to(DEVICE))
        train_correct_criteria4 += ((predictions_squeezed[3] >= 0.5) == criteria['Integrity of edges'].to(DEVICE)).type(torch.float).sum().item()

        loss_criteria5 = loss_function(predictions_squeezed[4], criteria['Straightness of edges'].to(DEVICE))
        train_correct_criteria5 += ((predictions_squeezed[4] >= 0.5) == criteria['Straightness of edges'].to(DEVICE)).type(torch.float).sum().item()

        loss_criteria6 = loss_function(predictions_squeezed[5], criteria['Size relative to type'].to(DEVICE))
        train_correct_criteria6 += ((predictions_squeezed[5] >= 0.5) == criteria['Size relative to type'].to(DEVICE)).type(torch.float).sum().item()

        loss_criteria7 = loss_function(predictions_squeezed[6], criteria['Conservation of color codin'].to(DEVICE))
        train_correct_criteria7 += ((predictions_squeezed[6] >= 0.5) == criteria['Conservation of color codin'].to(DEVICE)).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss_criteria1.backward()
        loss_criteria2.backward()
        loss_criteria3.backward()
        loss_criteria4.backward()
        loss_criteria5.backward()
        loss_criteria6.backward()
        loss_criteria7.backward()
        optimizer.step()
        
        total_loss_criteria1.append(loss_criteria1)
        total_loss_criteria2.append(loss_criteria2)
        total_loss_criteria3.append(loss_criteria3)
        total_loss_criteria4.append(loss_criteria4)
        total_loss_criteria5.append(loss_criteria5)
        total_loss_criteria6.append(loss_criteria6)
        total_loss_criteria7.append(loss_criteria7)
    accuracy = (train_correct_criteria1 + train_correct_criteria2 + train_correct_criteria3 + train_correct_criteria4 + train_correct_criteria5 +train_correct_criteria6 + train_correct_criteria7)/(len(dataset)*7)
    print(f"The accuracy of this batch during training is f{accuracy}")