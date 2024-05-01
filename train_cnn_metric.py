from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
from performance_evaluator import PerformancePredictor
from tqdm import tqdm
from torchvision.models import resnet50, resnet18
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
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

#PARAMETER INITIALIZATION
batch_size = 100
NUM_EPOCHS = 7
INIT_LR = 1e-5

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

#dataset = ImageDataset ('/home/sergio/Thesis_Sergio/inference_repo/blockgen_inference/outputs/new_model_200infsteps/random_images_test', 'labels_new_test.csv', transforms = transforms_regular)

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', 'shuffled_dataset/labels_101to200.csv', transforms=transforms_regular)
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv', transforms=transforms_regular)
dataset_1_aug = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', 'shuffled_dataset/labels_101to200.csv', transforms=augmentation_transforms)
dataset_2_aug = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv', transforms=augmentation_transforms)

dataset_regular = ConcatDataset([dataset_1, dataset_2])
dataset_augmented = ConcatDataset([dataset_1_aug, dataset_2_aug])
dataset = ConcatDataset([dataset_regular, dataset_augmented])

print(f'---------------------------------------------------')
print(f'\n\nThe length of the dataset is: {len(dataset)}')
print(f'---------------------------------------------------')

#train_loader = DataLoader(dataset, batch_size = batch_size)

#DECLARE THE MODEL
base_model = resnet18(pretrained = True)
for param in base_model.parameters():
	param.requires_grad = False

perf_evaluator_model = PerformancePredictor(base_model)
perf_evaluator_model = perf_evaluator_model.to(DEVICE)


    #initialize dictionaries to append the loss for each criteria during each fold
H_val = []
H_train = []
for i in range(k_folds):
    H_val.append({"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria3": [], "total_loss_criteria4": [], "total_loss_criteria5": [], "total_loss_criteria6": [], "total_loss_criteria7": [],
              "total_accuracy_criteria1": [], "total_accuracy_criteria2": [], "total_accuracy_criteria3": [], "total_accuracy_criteria4": [], "total_accuracy_criteria5": [], "total_accuracy_criteria6": [], "total_accuracy_criteria7": []})
    H_train.append({"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria3": [], "total_loss_criteria4": [], "total_loss_criteria5": [], "total_loss_criteria6": [], "total_loss_criteria7": [],
              "total_accuracy_criteria1": [], "total_accuracy_criteria2": [], "total_accuracy_criteria3": [], "total_accuracy_criteria4": [], "total_accuracy_criteria5": [], "total_accuracy_criteria6": [], "total_accuracy_criteria7": []})

#HERE WE ARE DECLARING THE BEST LOSS OBTAINED FOR EACH CRITERIA DURING THE CROSS VALIDATION PROCESS TO BE INITIALLY INFINITE 
best_loss_criteria1 = math.inf
best_loss_criteria2 = math.inf
best_loss_criteria3 = math.inf
best_loss_criteria4 = math.inf
best_loss_criteria5 = math.inf
best_loss_criteria6 = math.inf
best_loss_criteria7 = math.inf

val_steps = (len(dataset)*(1/k_folds)) // batch_size
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
    #reset weights of the network to train it from scratch in each fold
    perf_evaluator_model.apply(weights_init)
    #DECLARE THE OPTIMIZER
    optimizer = Adam(perf_evaluator_model.parameters(), lr=INIT_LR, weight_decay=1e-5) 
    
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
        
        val_correct_criteria1 = 0 #'Relative position and orientation between neighboring buildings'
        val_correct_criteria2 = 0 #'Position and orientation of buildings in relation to closest road/s'
        val_correct_criteria3 = 0 #'Building types in relation to underlying terrain type'
        val_correct_criteria4 = 0 #'Integrity of edges'
        val_correct_criteria5 = 0 #'Straightness of edges'
        val_correct_criteria6 = 0 #'Size relative to type'
        val_correct_criteria7 = 0 #'Conservation of color codin'
        
        total_val_loss_criteria1 = 0
        total_val_loss_criteria2 = 0
        total_val_loss_criteria3 = 0
        total_val_loss_criteria4 = 0
        total_val_loss_criteria5 = 0
        total_val_loss_criteria6 = 0
        total_val_loss_criteria7 = 0
        
        for (images, criteria) in tqdm(train_loader):
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

            loss_criteria7 = loss_function(predictions_squeezed[6], criteria['Conservation of color codi'].to(DEVICE))
            train_correct_criteria7 += ((predictions_squeezed[6] >= 0.5) == criteria['Conservation of color codi'].to(DEVICE)).type(torch.float).sum().item()

            optimizer.zero_grad()
            loss_criteria1.backward()
            loss_criteria2.backward()
            loss_criteria3.backward()
            loss_criteria4.backward()
            loss_criteria5.backward()
            loss_criteria6.backward()
            loss_criteria7.backward()
            optimizer.step()
            
            H_train[fold]["total_loss_criteria1"].append(loss_criteria1)
            H_train[fold]["total_loss_criteria2"].append(loss_criteria2)
            H_train[fold]["total_loss_criteria3"].append(loss_criteria3)
            H_train[fold]["total_loss_criteria4"].append(loss_criteria4)
            H_train[fold]["total_loss_criteria5"].append(loss_criteria5)
            H_train[fold]["total_loss_criteria6"].append(loss_criteria6)
            H_train[fold]["total_loss_criteria7"].append(loss_criteria7)
            
        for k in range(1, 8):
            var_name = "train_correct_criteria" + str(k)
            H_train[fold][f"total_accuracy_criteria{k}"].append(globals()[var_name]/(len(dataset)*((k_folds-1)/k_folds)))
            
        overall_accuracy = (train_correct_criteria1 + train_correct_criteria2 + train_correct_criteria3 + train_correct_criteria4 + train_correct_criteria5 +train_correct_criteria6 + train_correct_criteria7)/(len(dataset)*7)
        print(f"FOLD {fold}: The accuracy for epoch {e} during training is {overall_accuracy*100}%")
        with torch.no_grad():
			# set the model in evaluation mode
            perf_evaluator_model.eval()
   
            for (images, criteria) in tqdm(val_loader):
                
                images = images.to(DEVICE)
                predictions = perf_evaluator_model(images)
                predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions] 
                
                #we now calculate the losses

                loss_criteria1 = loss_function(predictions_squeezed[0], criteria['Relative position and orientation between neighboring buildings'].to(DEVICE))
                val_correct_criteria1 += ((predictions_squeezed[0] >= 0.5) == criteria['Relative position and orientation between neighboring buildings'].to(DEVICE)).type(torch.float).sum().item()
                
                loss_criteria2 = loss_function(predictions_squeezed[1], criteria['Position and orientation of buildings in relation to closest road/s'].to(DEVICE))
                val_correct_criteria2 += ((predictions_squeezed[1] >= 0.5) == criteria['Position and orientation of buildings in relation to closest road/s'].to(DEVICE)).type(torch.float).sum().item()
                
                loss_criteria3 = loss_function(predictions_squeezed[2], criteria['Building types in relation to underlying terrain type'].to(DEVICE))
                val_correct_criteria3 += ((predictions_squeezed[2] >= 0.5) == criteria['Building types in relation to underlying terrain type'].to(DEVICE)).type(torch.float).sum().item()

                loss_criteria4 = loss_function(predictions_squeezed[3], criteria['Integrity of edges'].to(DEVICE))
                val_correct_criteria4 += ((predictions_squeezed[3] >= 0.5) == criteria['Integrity of edges'].to(DEVICE)).type(torch.float).sum().item()

                loss_criteria5 = loss_function(predictions_squeezed[4], criteria['Straightness of edges'].to(DEVICE))
                val_correct_criteria5 += ((predictions_squeezed[4] >= 0.5) == criteria['Straightness of edges'].to(DEVICE)).type(torch.float).sum().item()

                loss_criteria6 = loss_function(predictions_squeezed[5], criteria['Size relative to type'].to(DEVICE))
                val_correct_criteria6 += ((predictions_squeezed[5] >= 0.5) == criteria['Size relative to type'].to(DEVICE)).type(torch.float).sum().item()

                loss_criteria7 = loss_function(predictions_squeezed[6], criteria['Conservation of color codi'].to(DEVICE))
                val_correct_criteria7 += ((predictions_squeezed[6] >= 0.5) == criteria['Conservation of color codi'].to(DEVICE)).type(torch.float).sum().item()
                
                total_val_loss_criteria1 += loss_criteria1
                total_val_loss_criteria2 += loss_criteria2
                total_val_loss_criteria3 += loss_criteria3
                total_val_loss_criteria4 += loss_criteria4
                total_val_loss_criteria5 += loss_criteria5
                total_val_loss_criteria6 += loss_criteria6
                total_val_loss_criteria7 += loss_criteria7
        for k in range(1, 8):
            var_name = "val_correct_criteria" + str(k)
            H_val[fold][f"total_accuracy_criteria{k}"].append(globals()[var_name]/(len(dataset)/k_folds))
        #no need to do the average validation loss, so much better to check each criteria's loss and save the best layer for each criteria
        avg_val_loss_criteria1 = total_val_loss_criteria1 / val_steps
        avg_val_loss_criteria2 = total_val_loss_criteria2 / val_steps
        avg_val_loss_criteria3 = total_val_loss_criteria3 / val_steps
        avg_val_loss_criteria4 = total_val_loss_criteria4 / val_steps
        avg_val_loss_criteria5 = total_val_loss_criteria5 / val_steps
        avg_val_loss_criteria6 = total_val_loss_criteria6 / val_steps
        avg_val_loss_criteria7 = total_val_loss_criteria7 / val_steps
        for k in range(1, 8):
            var_name = "val_correct_criteria" + str(k)
            H_val[fold][f"total_loss_criteria{k}"].append(globals()[var_name])
        
        if e == NUM_EPOCHS - 1:
            for i in range (1, 8):
                avg_val_loss = "avg_val_loss_criteria" + str(i)
                best_loss = "best_loss_criteria" + str(i)
                if "best_model_criteria" + str(i) not in globals():
                    globals()["best_model_criteria" +str(i)] = perf_evaluator_model
                    best_performing_folds[i-1] = copy.deepcopy(fold)
                else:
                    if globals()[avg_val_loss] < globals()[best_loss]:
                        best_model = globals()["best_model_criteria" +str(i)]
                        globals()[best_loss] = globals()[avg_val_loss]
                        print(f'Average loss during validation for the criteria {i}: {globals()[avg_val_loss]}')
                        best_model = perf_evaluator_model
                        best_performing_folds[i-1] = copy.deepcopy(fold)
print("[INFO] saving performance evaluator model...")
perf_evaluator_model.criteria1 = best_model_criteria1.criteria1   
perf_evaluator_model.criteria2 = best_model_criteria2.criteria2        
perf_evaluator_model.criteria3 = best_model_criteria3.criteria3        
perf_evaluator_model.criteria4 = best_model_criteria4.criteria4        
perf_evaluator_model.criteria5 = best_model_criteria5.criteria5        
perf_evaluator_model.criteria6 = best_model_criteria6.criteria6        
perf_evaluator_model.criteria7 = best_model_criteria7.criteria7      
  
# Define the directory where you want to save the model
output_dir = "output model"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model path
model_path = os.path.join(output_dir, "performance_evaluator.pth")

torch.save(perf_evaluator_model, model_path)

#output folder initialization
save_folder = "output plots/training_accuracies_validation"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
criteria_names = ["Relative position and orientation between neighboring buildings", 
                  "Position and orientation of buildings in relation to closest roads", 
                  "Building types in relation to underlying terrain type", 
                  "Integrity of edges", 
                  "Straightness of edges", 
                  "Size relative to type", 
                  "Conservation of color coding"]

for i in range(1, 8):
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
    
"""                
total_loss_criteria1 = torch.tensor(total_loss_criteria1, device = 'cpu')
plt.figure()
plt.plot(total_loss_criteria1)
plt.savefig("dummy_name.png")
"""
