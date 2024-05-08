from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split
from Custom_Trainer import CustomTrainer
from Evaluate_Model import evaluate_model
from torch.utils.data import ConcatDataset
from collections import defaultdict
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
target_size = (768, 768)

transforms_regular = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/First100', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/labels_First100.csv', transforms = transforms_regular)
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/labels_101to200.csv', transforms = transforms_regular)
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/201to300', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/labels_201to300.csv', transforms = transforms_regular)
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/labels_301to400.csv', transforms = transforms_regular)
upsample_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings/upsampled_labels.csv', transforms = transforms_regular)
upsample_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings1', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings1/upsampled_labels.csv', transforms = transforms_regular)
upsample_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings2', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings2/upsampled_labels.csv', transforms = transforms_regular)
upsample_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings3', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings3/upsampled_labels.csv', transforms = transforms_regular)
upsample_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads/upsampled_labels.csv', transforms = transforms_regular)
upsample_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads2', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads2/upsampled_labels.csv', transforms = transforms_regular)
upsample_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads3', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads3/upsampled_labels.csv', transforms = transforms_regular)
upsample_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads4', '/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads4/upsampled_labels.csv', transforms = transforms_regular)
dataset_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/First25', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_First25.csv')
dataset_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/26to50', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_26to50.csv')
dataset_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/51to75', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_51to75.csv')
dataset_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/76to100', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_76to100.csv')

dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8])

test_ratio = 0.2
num_total = len(dataset)
num_test = int(test_ratio * num_total)
num_train = num_total - num_test


batch_size = 15
# Split the dataset into training, validation, and test sets
train_dataset_aux, test_dataset = random_split(dataset, [num_train, num_test])
num_total_train = len(train_dataset_aux)
val_ratio = 0.2
num_train_v = int(val_ratio * num_total)
num_train_t = num_total_train - num_train_v
train_dataset, val_dataset = random_split(train_dataset_aux, [num_train_t, num_train_v])

#done this weird split in order to have the exact same number of elements as in the other multihead trainings for training, validation and testing

# Create DataLoader for training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create DataLoader for test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize a dictionary of counters to store the frequency of each label for each criterion
label_frequencies = defaultdict(lambda: defaultdict(int))

# Iterate over the dataset to count the frequency of each label for each criterion
for image, labels_dict in dataset:
    for criterion, labels in labels_dict.items():
        for label, freq in enumerate(labels):
            label_frequencies[criterion][label] += freq.item()

# Print the label frequencies for each criterion
print("Class frequencies per criterion:")
for criterion, frequencies in label_frequencies.items():
    print(f"Criterion: {criterion}")
    for label, frequency in frequencies.items():
        print(f"Label {label}: {frequency}")
    print()
    
# Calculate the total number of samples in the dataset
total_samples = len(dataset)

# Convert frequencies to tensors if they are not already tensors
for criterion, frequencies in label_frequencies.items():
    for label, freq in frequencies.items():
        if not isinstance(freq, torch.Tensor):
            frequencies[label] = torch.tensor(freq)

# Calculate class weights for each criterion based on the frequency of each label
class_weights = {}
for criterion, frequencies in label_frequencies.items():
    class_weights[criterion] = [total_samples / (freq.item() * 2) for freq in frequencies.values()]

# Print the class weights for each criterion
print("Class weights per criterion:")
for criterion, weights in class_weights.items():
    print(f"{criterion}: {weights}")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-4 no Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_LR1e_4 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_LR1e_4 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_LR1e_4 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_LR1e_4 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_LR1e_4/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_LR1e_4/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_LR1e_4/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_LR1e_4/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_4"

evaluate_model(test_loader, model_paths, output_folder_test)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-3 no Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_LR1e_3 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_LR1e_3 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_LR1e_3 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_LR1e_3 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_LR1e_3/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_LR1e_3/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_LR1e_3/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_LR1e_3/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_3"

evaluate_model(test_loader, model_paths, output_folder_test)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-2 no Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_LR1e_2 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_LR1e_2 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_LR1e_2 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_LR1e_2 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_LR1e_2/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_LR1e_2/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_LR1e_2/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_LR1e_2/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_2"

evaluate_model(test_loader, model_paths, output_folder_test)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-4 Weighted classes------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader, class_weights)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_Class_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_Class_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_4', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_Class_LR1e_4/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_Class_LR1e_4/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_4/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_4/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_4_CW"

evaluate_model(test_loader, model_paths, output_folder_test)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-3 Weighted classes------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader, class_weights)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_Class_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_Class_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_3', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_Class_LR1e_3/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_Class_LR1e_3/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_3/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_3/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_3_CW"

evaluate_model(test_loader, model_paths, output_folder_test)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#-----------------------------------LR1e-2 Weighted classes------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

trainer = CustomTrainer(train_loader, val_loader, class_weights)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_Class_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 30)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_Class_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 30)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Integrity of edges', NUM_EPOCHS = 30)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_Class_LR1e_2', 'Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Straightness of edges', NUM_EPOCHS = 30)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_Class_LR1e_2/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_Class_LR1e_2/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_2/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_Class_LR1e_2/Straightness of edges.pth'
}


output_folder_test = "TEST_ResNet18_LR1e_2_CW"

evaluate_model(test_loader, model_paths, output_folder_test)
