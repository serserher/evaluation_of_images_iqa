from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
from performance_evaluator import PerformancePredictor
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import os
import copy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.data.dataset import random_split


class TrainingPipeline:
    def __init__(self, train_dataset, val_dataset, backbone_model, batch_size=32, k_folds=5, num_epochs=10, init_lr=0.001, class_weights = None, train_backbone = True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.backbone_model = backbone_model
        self.perf_evaluator_model = PerformancePredictor(backbone_model, 0.5)
        self.batch_size = batch_size
        #self.k_folds = k_folds
        #self.kfold = KFold(n_splits=k_folds, shuffle=True)
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.class_weights = class_weights
        self.train_backbone = train_backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
                

        self.optimizer = None
        self.scheduler = None
        self.H_val = {"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria3": [],
                      "total_loss_criteria4": [], "total_accuracy_criteria1": [], "total_accuracy_criteria2": [],
                      "total_accuracy_criteria3": [], "total_accuracy_criteria4": []}
        self.H_train = {"total_loss_criteria1": [], "total_loss_criteria2": [], "total_loss_criteria3": [],
                        "total_loss_criteria4": [], "total_accuracy_criteria1": [], "total_accuracy_criteria2": [],
                        "total_accuracy_criteria3": [], "total_accuracy_criteria4": []}
        #self.best_performing_folds = [0, 0, 0, 0, 0, 0, 0]
        self.best_models = {}
        
    def linear_lr_scheduler(self, epoch):
        lr = self.init_lr * (1 - epoch / self.num_epochs)
        return lr
    
    def train(self):

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # Clone the backbone model

        self.perf_evaluator_model = self.perf_evaluator_model.to(self.device)

        self.optimizer = Adam(self.perf_evaluator_model.parameters(), lr=self.init_lr, weight_decay=1e-5)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.linear_lr_scheduler(epoch))
        self.criteria_names = ['Relative position and orientation between neighboring buildings', 'Position and orientation of buildings in relation to closest road/s', 'Integrity of edges', 'Straightness of edges']
        if self.class_weights:
            loss_functions = []
            for i in range(4):
                loss_functions.append(CrossEntropyLoss(weight=torch.tensor(self.class_weights[self.criteria_names[i]], device=self.device)))
        best_loss = [math.inf, math.inf, math.inf, math.inf]
        best_loss_total = math.inf
        no_improvement = 0
        backbone_training = False
        total_loss_array = []
        for e in tqdm(range(self.num_epochs)):
            print(f'Epoch {e} of the training starting')

            # Training loop
            self.perf_evaluator_model.train()
            train_correct = [0, 0, 0, 0]
            total_loss = [0, 0, 0, 0]

            for images, criteria in tqdm(train_loader):
                images = images.to(self.device)
                predictions = self.perf_evaluator_model(images)

                self.optimizer.zero_grad()

                losses = []
                for j, pred in enumerate(predictions):
                    if not self.class_weights:
                        loss = self.criterion(torch.squeeze(pred, dim=1), criteria[self.criteria_names[j]].float().to(self.device))
                    else:
                        loss = loss_functions[j](torch.squeeze(pred, dim=1), criteria[self.criteria_names[j]].float().to(self.device))
                    if self.train_backbone:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    losses.append(loss.item())
                    train_correct[j] += ((pred > 0.5).float() == criteria[self.criteria_names[j]].float().to(self.device)).all(dim=1).sum().item()
                    total_loss[j] += loss.item()
                
                self.optimizer.step()

            for j in range(4):
                self.H_train[f"total_loss_criteria{j+1}"].append(total_loss[j] / len(train_loader))

            overall_accuracy_train = 0
            for j in range(4):
                self.H_train[f"total_accuracy_criteria{j+1}"].append(train_correct[j] / len(train_loader.dataset))
                overall_accuracy_train += train_correct[j] / len(train_loader.dataset) / 4
            print(f"The overall accuracy obtained during training in epoch {e} is: {overall_accuracy_train*100}%")

            # Validation loop
            self.perf_evaluator_model.eval()
            val_correct = [0, 0, 0, 0]
            val_total_loss = [0, 0, 0, 0]
            total_loss = 0
            for images, criteria in val_loader:
                images = images.to(self.device)
                with torch.no_grad():
                    predictions = self.perf_evaluator_model(images)

                for j, pred in enumerate(predictions):
                    val_loss = self.criterion(torch.squeeze(pred, dim=1), criteria[self.criteria_names[j]].float().to(self.device))
                    val_correct[j] += ((pred > 0.5).float() == criteria[self.criteria_names[j]].float().to(self.device)).all(dim=1).sum().item()
                    val_total_loss[j] += val_loss.item()

            overall_accuracy_val = 0
            for j in range(4):
                total_loss += val_total_loss[j]
                if f"best_model_criteria{j+1}" not in self.best_models:
                    self.best_models[f"best_model_criteria{j+1}"] = copy.deepcopy(self.perf_evaluator_model)
                else:
                    if val_total_loss[j] < best_loss[j]:
                        print(f"Best new weights found for the criteria {self.criteria_names[j]}, the old loss value is {best_loss[j]} and the new one is  {val_total_loss[j]}!")
                        best_loss[j] = copy.deepcopy(val_total_loss[j])
                        self.best_models[f"best_model_criteria{j+1}"] = copy.deepcopy(self.perf_evaluator_model)
                        
                self.H_val[f"total_loss_criteria{j+1}"].append(val_total_loss[j] / len(val_loader))
                self.H_val[f"total_accuracy_criteria{j+1}"].append(val_correct[j] / len(val_loader.dataset))
                overall_accuracy_val += val_correct[j] / len(val_loader.dataset) / 4
            total_loss_array.append(total_loss)
            
            if total_loss > 0.99999*best_loss_total and not backbone_training:
                print(f'The total loss obtained in validation in epoch {e} is :{total_loss}, the best one obtained so far is: {best_loss_total}')
                no_improvement  += 1
                if no_improvement >=2:
                    backbone_training = True
                    for param in self.perf_evaluator_model.BaseModel.parameters():
                        param.requires_grad = True
                    print("\n\n\n\nThe backbone is also being trained now \n\n\n\n")
                    
            if e > 20:
                if (int(total_loss_array[-1] > 0.9999*total_loss_array[-2]) + int(total_loss_array[-2] > 0.9999*total_loss_array[-3]) + int(total_loss_array[-3] > 0.9999*total_loss_array[-4]) + int(total_loss_array[-4] > 0.9999*total_loss_array[-5]) + int(total_loss_array[-5] > 0.9999*total_loss_array[-6]) + int(total_loss_array[-6] > 0.9999*total_loss_array[-7])) > 3:
                    return       
                
            if total_loss < best_loss_total:
                best_loss_total = total_loss
        


            print(f"The overall accuracy obtained during validation in epoch {e} is: {overall_accuracy_val*100}%")

            # Update learning rate
            self.scheduler.step()

        # Save best performing models
        
    def save_models(self, output_dir):
        #self.perf_evaluator_model_global.to(self.device)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, model in self.best_models.items():
            model_path = os.path.join(output_dir, f"full_model.pth")
            if key == 'best_model_criteria1':
                self.perf_evaluator_model.criteria1_conv = model.criteria1_conv
                self.perf_evaluator_model.criteria1_lin = model.criteria1_lin
            if key == 'best_model_criteria2':
                self.perf_evaluator_model.criteria2_conv = model.criteria2_conv
                self.perf_evaluator_model.criteria2_lin = model.criteria2_lin
            if key == 'best_model_criteria3':
                self.perf_evaluator_model.criteria3_conv = model.criteria3_conv
                self.perf_evaluator_model.criteria3_lin = model.criteria3_lin
            if key == 'best_model_criteria4':
                self.perf_evaluator_model.criteria4_conv = model.criteria4_conv
                self.perf_evaluator_model.criteria4_lin = model.criteria4_lin
        torch.save(self.perf_evaluator_model, model_path)

    def plot_metrics(self, save_folder, criteria_names = ['Relative position and orientation between neighboring buildings', 'Position and orientation of buildings in relation to closest roads', 'Integrity of edges', 'Straightness of edges']):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for j, name in enumerate(criteria_names):
            plt.plot(range(1, len(self.H_train[f"total_accuracy_criteria{j+1}"]) + 1),self.H_train[f"total_accuracy_criteria{j+1}"], label='Training Accuracy')
            plt.plot(range(1, len(self.H_val[f"total_accuracy_criteria{j+1}"]) + 1),self.H_val[f"total_accuracy_criteria{j+1}"], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy Plot for the criterion: {name}')
            plt.legend()
            plt.savefig(os.path.join(save_folder, f'Comparison_accuracies_{name}.png'))
            plt.close()

            plt.plot(range(1, len(self.H_train[f"total_loss_criteria{j+1}"]) + 1),self.H_train[f"total_loss_criteria{j+1}"], label='Training Loss')
            plt.plot(range(1, len(self.H_val[f"total_loss_criteria{j+1}"]) + 1),self.H_val[f"total_loss_criteria{j+1}"], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Plot for the criterion: {name}')
            plt.legend()
            plt.savefig(os.path.join(save_folder, f'Comparison_losses_{name}.png'))
            plt.close()
            


