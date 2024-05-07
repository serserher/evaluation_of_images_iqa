from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
from performance_evaluator import PerformancePredictor
from tqdm import tqdm
from torchvision.models import resnet50, resnet18, ResNet18_Weights, mobilenet_v2
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import math
import os
from torch.utils.data import ConcatDataset
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

#PARAMETER INITIALIZATION
batch_size = 15
NUM_EPOCHS = 20
INIT_LR = 1e-4

# DEFINE LINEAR LEARNING RATE SCHEDULER FUNCTION
def linear_lr_scheduler(epoch):
    lr = INIT_LR * (1 - epoch / NUM_EPOCHS)
    return lr
        
#TO ADDRESS THE FEW DATA SAMPLES WE WILL USE CROSS VALIDATION 
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
#WEIGHT INITIALIZATION AFTER EACH FOLD
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        
loss_function = CrossEntropyLoss()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
target_size = (768, 768)

transforms_regular = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/First100', 'shuffled_dataset/labels_First100.csv', transforms = transforms_regular)
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', 'shuffled_dataset/labels_101to200.csv', transforms = transforms_regular)
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/201to300', 'shuffled_dataset/labels_201to300.csv', transforms = transforms_regular)
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv', transforms = transforms_regular)
upsample_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings', 'upsampled/PositioningBuildings/upsampled_labels.csv', transforms = transforms_regular)
upsample_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings1', 'upsampled/PositioningBuildings1/upsampled_labels.csv', transforms = transforms_regular)
upsample_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings2', 'upsampled/PositioningBuildings2/upsampled_labels.csv', transforms = transforms_regular)
upsample_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings3', 'upsampled/PositioningBuildings3/upsampled_labels.csv', transforms = transforms_regular)
upsample_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads', 'upsampled/PositioningRoads/upsampled_labels.csv', transforms = transforms_regular)
upsample_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads2', 'upsampled/PositioningRoads2/upsampled_labels.csv', transforms = transforms_regular)
upsample_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads3', 'upsampled/PositioningRoads3/upsampled_labels.csv', transforms = transforms_regular)
upsample_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads4', 'upsampled/PositioningRoads4/upsampled_labels.csv', transforms = transforms_regular)

dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8])

print(f'---------------------------------------------------')
print(f'\n\nThe length of the dataset is: {len(dataset)}')
print(f'---------------------------------------------------')

#DECLARE THE MODEL
base_model = mobilenet_v2(pretrained=True)
for param in base_model.parameters():
	param.requires_grad = False
print(base_model)
perf_evaluator_model = PerformancePredictor(base_model, 0.5)
perf_evaluator_model = perf_evaluator_model.to(DEVICE)


#initialize dictionaries to append the loss for each criteria during each fold
H_val = []
H_train = []

for i in range(k_folds):
    H_val.append({"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria4": [], "total_loss_criteria5": [], 
                  "total_accuracy_criteria1": [], "total_accuracy_criteria2": [], "total_accuracy_criteria4": [], "total_accuracy_criteria5": []})
    H_train.append({"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria4": [], "total_loss_criteria5": [], 
                  "total_accuracy_criteria1": [], "total_accuracy_criteria2": [], "total_accuracy_criteria4": [], "total_accuracy_criteria5": []})


#HERE WE ARE DECLARING THE BEST LOSS OBTAINED FOR EACH CRITERIA DURING THE CROSS VALIDATION PROCESS TO BE INITIALLY INFINITE 
best_loss_criteria1 = math.inf
best_loss_criteria2 = math.inf
best_loss_criteria4 = math.inf
best_loss_criteria5 = math.inf


val_steps = (len(dataset)*(1/k_folds)) // batch_size if ((len(dataset)*(1/k_folds)) // batch_size > 1) else 1
best_performing_folds = [0,0,0,0,0,0,0]
for fold, (train_ids, val_ids) in enumerate (kfold.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_loader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=val_subsampler)
    
    perf_evaluator_model.criteria1.apply(weights_init)
    perf_evaluator_model.criteria2.apply(weights_init)
    perf_evaluator_model.criteria4.apply(weights_init)
    perf_evaluator_model.criteria5.apply(weights_init)

    
    #DECLARE THE OPTIMIZER
    optimizer = Adam(perf_evaluator_model.parameters(), lr=INIT_LR, weight_decay=1e-5) 
    # Add linear learning rate scheduler after initializing the optimizer
    # INITIALIZE LEARNING RATE SCHEDULER
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: linear_lr_scheduler(epoch))

    for e in tqdm(range(NUM_EPOCHS)):
        perf_evaluator_model.train()
        
        
        perf_evaluator_model.train()
        
        #initialize the number of correct evaluations during the training
        train_correct_criteria1 = 0 #'Relative position and orientation between neighboring buildings'
        train_correct_criteria2 = 0 #'Position and orientation of buildings in relation to closest road/s'
        train_correct_criteria4 = 0 #'Integrity of edges'
        train_correct_criteria5 = 0 #'Straightness of edges'

        
        val_correct_criteria1 = 0 #'Relative position and orientation between neighboring buildings'
        val_correct_criteria2 = 0 #'Position and orientation of buildings in relation to closest road/s'
        val_correct_criteria4 = 0 #'Integrity of edges'
        val_correct_criteria5 = 0 #'Straightness of edges'

        
        total_val_loss_criteria1 = 0
        total_val_loss_criteria2 = 0
        total_val_loss_criteria4 = 0
        total_val_loss_criteria5 = 0


        for (images, criteria) in tqdm(train_loader):
            images = images.to(DEVICE)
            predictions = perf_evaluator_model(images)
            predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions] 
            #we now calculate the losses
            loss_criteria1 = loss_function(predictions_squeezed[0], criteria['Relative position and orientation between neighboring buildings'].float().to(DEVICE))
            train_correct_criteria1 += ((predictions_squeezed[0] > 0.5).float() == criteria['Relative position and orientation between neighboring buildings'].float().to(DEVICE)).all(dim=1).sum().item()

            loss_criteria2 = loss_function(predictions_squeezed[1], criteria['Position and orientation of buildings in relation to closest road/s'].float().to(DEVICE))
            train_correct_criteria2 += ((predictions_squeezed[1] > 0.5).float() == criteria['Position and orientation of buildings in relation to closest road/s'].float().to(DEVICE)).all(dim=1).sum().item()
            
            loss_criteria4 = loss_function(predictions_squeezed[2], criteria['Integrity of edges'].float().to(DEVICE))
            train_correct_criteria4 += ((predictions_squeezed[2] > 0.5).float() == criteria['Integrity of edges'].float().to(DEVICE)).all(dim=1).sum().item()

            loss_criteria5 = loss_function(predictions_squeezed[3], criteria['Straightness of edges'].float().to(DEVICE))
            train_correct_criteria5 += ((predictions_squeezed[3] > 0.5).float() == criteria['Straightness of edges'].float().to(DEVICE)).all(dim=1).sum().item()


            optimizer.zero_grad()
            loss_criteria1.backward()
            loss_criteria2.backward()
            loss_criteria4.backward()
            loss_criteria5.backward()
            
            optimizer.step()
            
            H_train[fold]["total_loss_criteria1"].append(loss_criteria1)
            H_train[fold]["total_loss_criteria2"].append(loss_criteria2)
            H_train[fold]["total_loss_criteria4"].append(loss_criteria4)
            H_train[fold]["total_loss_criteria5"].append(loss_criteria5)

            
        for k in range(1, 5):
            if k == 1:
                o = 1
            elif k == 2:
                o = 2
            elif k == 3:
                o = 4
            else:
                o = 5
            var_name = "train_correct_criteria" + str(o)
            H_train[fold][f"total_accuracy_criteria{o}"].append(globals()[var_name]/(len(dataset)*((k_folds-1)/k_folds)))
            
        overall_accuracy = (train_correct_criteria1 + train_correct_criteria2 + train_correct_criteria4 + train_correct_criteria5)/(len(train_loader)*batch_size*4)
        print(f"FOLD {fold}: The accuracy for epoch {e} during training is {overall_accuracy*100}%")
        with torch.no_grad():
			# set the model in evaluation mode
            perf_evaluator_model.eval()
   
            for (images, criteria) in val_loader:
                
                images = images.to(DEVICE)
                predictions = perf_evaluator_model(images)
                predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions] 
                
                #we now calculate the losses

                loss_criteria1 = loss_function(predictions_squeezed[0], criteria['Relative position and orientation between neighboring buildings'].float().to(DEVICE))
                val_correct_criteria1 += ((predictions_squeezed[0] > 0.5).float() == criteria['Relative position and orientation between neighboring buildings'].float().to(DEVICE)).all(dim=1).sum().item()

                loss_criteria2 = loss_function(predictions_squeezed[1], criteria['Position and orientation of buildings in relation to closest road/s'].float().to(DEVICE))
                val_correct_criteria2 += ((predictions_squeezed[1] > 0.5).float() == criteria['Position and orientation of buildings in relation to closest road/s'].float().to(DEVICE)).all(dim=1).sum().item()

                loss_criteria4 = loss_function(predictions_squeezed[2], criteria['Integrity of edges'].float().to(DEVICE))
                val_correct_criteria4 += ((predictions_squeezed[2] > 0.5).float() == criteria['Integrity of edges'].float().to(DEVICE)).all(dim=1).sum().item()

                loss_criteria5 = loss_function(predictions_squeezed[3], criteria['Straightness of edges'].float().to(DEVICE))
                val_correct_criteria5 += ((predictions_squeezed[3] > 0.5).float() == criteria['Straightness of edges'].float().to(DEVICE)).all(dim=1).sum().item()

                total_val_loss_criteria1 += loss_criteria1
                total_val_loss_criteria2 += loss_criteria2
                total_val_loss_criteria4 += loss_criteria4
                total_val_loss_criteria5 += loss_criteria5

        for k in range(1, 5):
            if k == 1:
                o = 1
            elif k == 2:
                o = 2
            elif k == 3:
                o = 4
            else:
                o = 5
            var_name = "val_correct_criteria" + str(o)
            H_val[fold][f"total_accuracy_criteria{o}"].append(globals()[var_name]/(len(dataset)/k_folds))
        #no need to do the average validation loss, so much better to check each criteria's loss and save the best layer for each criteria
        avg_val_loss_criteria1 = total_val_loss_criteria1 / val_steps
        avg_val_loss_criteria2 = total_val_loss_criteria2 / val_steps
        avg_val_loss_criteria4 = total_val_loss_criteria4 / val_steps
        avg_val_loss_criteria5 = total_val_loss_criteria5 / val_steps

        for k in range(1, 5):
            if k == 1:
                o = 1
            elif k == 2:
                o = 2
            elif k == 3:
                o = 4
            else:
                o = 5
            var_name = "val_correct_criteria" + str(o)
            H_val[fold][f"total_loss_criteria{o}"].append(globals()[var_name])
        
        for k in range (1, 5):
            if k == 1:
                i = 1
            elif k == 2:
                i = 2
            elif k == 3:
                i = 4
            else:
                i = 5
            avg_val_loss = "avg_val_loss_criteria" + str(i)
            best_loss = "best_loss_criteria" + str(i)
            if "best_model_criteria" + str(i) not in globals():
                globals()["best_model_criteria" +str(i)] = copy.deepcopy(perf_evaluator_model)
                best_performing_folds[i-1] = copy.deepcopy(fold)
            else:
                if globals()[avg_val_loss] < globals()[best_loss]:
                    best_model = globals()["best_model_criteria" +str(i)]
                    globals()[best_loss] = globals()[avg_val_loss]
                    print(f'Average loss during validation for the criteria {i}: {globals()[avg_val_loss]}')
                    best_model = copy.deepcopy(perf_evaluator_model)
                    best_performing_folds[i-1] = copy.deepcopy(fold)
        # Call the learning rate scheduler to update the learning rate
        scheduler.step()
        
print("[INFO] saving performance evaluator model...")
perf_evaluator_model.criteria1 = best_model_criteria1.criteria1   
perf_evaluator_model.criteria2 = best_model_criteria2.criteria2                
perf_evaluator_model.criteria4 = best_model_criteria4.criteria4        
perf_evaluator_model.criteria5 = best_model_criteria5.criteria5            
  
# Define the directory where you want to save the model
output_dir = "output model Resnet 50"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model path
model_path = os.path.join(output_dir, "performance_evaluator_LR1e4_Dropout5_MoreLayers.pth")

torch.save(perf_evaluator_model, model_path)

#output folder for plots initialization
save_folder = "output plots/training_ResNet50_LR1e4_More_Complex_Classifying_Heads"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
criteria_names = ["Relative position and orientation between neighboring buildings", 
                  "Position and orientation of buildings in relation to closest roads", 
                  "Building types in relation to underlying terrain type", 
                  "Integrity of edges", 
                  "Straightness of edges", 
                  "Size relative to type", 
                  "Conservation of color coding"]

for k in range(1, 5):
    if k == 1:
        i = 1
    elif k == 2:
        i = 2
    elif k == 3:
        i = 4
    else:
        i = 5
    plt.plot(H_val[best_performing_folds[i-1]][f"total_accuracy_criteria{i}"], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Plot for the criterion: {criteria_names[i-1]}')
    plt.legend()
    plt.savefig(os.path.join(save_folder, f'best_performing_folds_accuracies_{criteria_names[i-1]}.png'))
    plt.close()
    #now the loss
    plt.plot(H_val[best_performing_folds[i-1]][f"total_loss_criteria{i}"], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Plot for the criterion: {criteria_names[i-1]}')
    plt.legend()
    plt.savefig(os.path.join(save_folder, f'best_performing_folds_losses_{criteria_names[i-1]}.png'))
    plt.close()

