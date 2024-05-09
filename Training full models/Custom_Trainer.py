import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomTrainer:
    def __init__(self, train_loader, val_loader, class_weights = None, device="cuda"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.device = device
        self.scaler = GradScaler()

    def train(self, criterion, model_path, figures_path, NUM_EPOCHS=10, INIT_LR=1e-4):
        print("-----------------------------------------------------------------------")
        print(f'Training a model to evaluate the criterion: {criterion}')
        print("-----------------------------------------------------------------------")

        if criterion == 'Position and orientation of buildings in relation to closest road/s':
            output_name = 'Position and orientation of buildings in relation to closest roads'
        else:
            output_name = criterion
            
        perf_evaluator_model = resnet18(pretrained=True)
        last_in_features_count = perf_evaluator_model.fc.in_features
        perf_evaluator_model.fc = nn.Linear(last_in_features_count, 2)
        perf_evaluator_model = perf_evaluator_model.to(self.device)

        optimizer = Adam(perf_evaluator_model.parameters(), lr=INIT_LR, weight_decay=1e-5) 
        if self.class_weights is None:
            loss_function = CrossEntropyLoss()
        else:
            loss_function = CrossEntropyLoss(weight=torch.tensor(self.class_weights[criterion], device=self.device))

        total_loss = []
        accuracy_during_training = []
        
        val_total_loss = []
        val_accuracy_during_training = []
        best_loss = math.inf

        for e in tqdm(range(NUM_EPOCHS)):
            perf_evaluator_model.train()
            train_correct = 0
            total_train = 0
            total_loss_val = 0
            val_correct = 0
            total_val = 0
            for (images, criteria) in tqdm(self.train_loader):
                images = images.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass and loss computation within autocast context
                with autocast():
                    predictions = perf_evaluator_model(images)
                    loss = loss_function(predictions, criteria[criterion].float().to(self.device))
                    total_loss.append(loss)
                    
                # Scale the loss and perform backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Track correct predictions and total samples for training accuracy calculation
                train_correct += ((predictions > 0.5).float() == criteria[criterion].float().to(self.device)).all(dim=1).sum().item()
                total_train += criteria[criterion].size(0)       

            # Calculate training accuracy
            training_acc = train_correct / total_train
            accuracy_during_training.append(training_acc)
            print(f"Training accuracy in epoch {e}: {training_acc}")

            # Validation loop with autocast
            for (images, criteria) in tqdm(self.val_loader):
                images = images.to(self.device)

                # Perform forward pass within autocast context
                with torch.no_grad(), autocast():
                    predictions = perf_evaluator_model(images)
                    val_loss = loss_function(predictions, criteria[criterion].float().to(self.device))
                    val_total_loss.append(val_loss)
                    total_loss_val += val_loss
                    val_correct += ((predictions > 0.5).float() == criteria[criterion].float().to(self.device)).all(dim=1).sum().item()
                    total_val += criteria[criterion].size(0)
            # Save the model if it performs better on the validation set
            if total_loss_val < best_loss:
                print(f"the validation loss for the newly found best performing model: {val_loss}")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                output_path= os.path.join(model_path, f'{output_name}.pth')
                torch.save(perf_evaluator_model, output_path)
            val_acc = val_correct / total_val
            val_accuracy_during_training.append(val_acc)
            print(f"the validation accuracy in the last iteration of epoch {e}: {val_acc}")
            if e > 20:
                if (int(val_total_loss[-1] > 0.9999*val_total_loss[-2]) + int(val_total_loss[-2] > 0.9999*val_total_loss[-3]) + int(val_total_loss[-3] > 0.9999*val_total_loss[-4]) + int(val_total_loss[-4] > 0.9999*val_total_loss[-5]) + int(val_total_loss[-5] > 0.9999*val_total_loss[-6]) + int(val_total_loss[-6] > 0.9999*val_total_loss[-7])) > 3:
                    break
        # Save figures for total loss and validation total loss
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)
        plt.figure(figsize=(10, 5))
        total_loss_tensor = torch.tensor(total_loss)  # Convert list to tensor
        plt.plot(total_loss_tensor.cpu().numpy(), label='Total Loss')
        val_total_loss_tensor = torch.tensor(val_total_loss)  # Convert list to tensor
        plt.plot(val_total_loss_tensor.cpu().numpy(), label='Validation Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(f'Loss comparison {output_name}')
        plt.legend()
        plt.savefig(os.path.join(figures_path, f'{output_name}_loss_comparison.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_during_training, label='Training accuracy')
        plt.plot(val_accuracy_during_training, label='Validation accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy comparison {output_name}')
        plt.legend()
        plt.savefig(os.path.join(figures_path, f'{output_name}_accuracies_evolution.png'))
        plt.close()