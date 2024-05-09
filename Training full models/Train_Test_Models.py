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

augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

transforms_regular = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/First75', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/First75/labels_First75.csv', transforms = transforms_regular)
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/101to175', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/101to175/labels_101to175.csv', transforms = transforms_regular)
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/201to274', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/201to274/labels_201to275.csv', transforms = transforms_regular)
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/301to375', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/301to375/labels_301to375.csv', transforms = transforms_regular)
upsample_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings1/upsampled_labels.csv', transforms = transforms_regular)
upsample_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings2/upsampled_labels.csv', transforms = augmentation_transforms)
upsample_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings3', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings3/upsampled_labels.csv', transforms = augmentation_transforms)
upsample_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings4', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings4/upsampled_labels.csv', transforms = transforms_regular)
upsample_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads1/upsampled_labels.csv', transforms = augmentation_transforms)
upsample_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads2/upsampled_labels.csv', transforms = transforms_regular)
upsample_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads3', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads3/upsampled_labels.csv', transforms = augmentation_transforms)
upsample_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads4', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads4/upsampled_labels.csv', transforms = transforms_regular)
upsample_9 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges1/upsampled_labels.csv', transforms = transforms_regular)
upsample_10 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges2/upsampled_labels.csv', transforms = augmentation_transforms)

train_dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8, upsample_9, upsample_10])


dataset_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/76to100', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/76to100/labels_76to100.csv', transforms = transforms_regular)
dataset_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/176to200', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/176to200/labels_176to200.csv', transforms = transforms_regular)
dataset_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/275to300', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/275to300/labels_275to300.csv', transforms = transforms_regular)
dataset_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/376to400', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/376to400/labels_376to400.csv', transforms = transforms_regular)
dataset_9 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/First25', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_First25.csv', transforms = transforms_regular)
dataset_10 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/26to50', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_26to50.csv', transforms = transforms_regular)
dataset_11 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/51to75', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_51to75.csv', transforms = transforms_regular)
dataset_12 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/76to100', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_76to100.csv', transforms = transforms_regular)

val_test_dataset = ConcatDataset([dataset_5, dataset_6, dataset_7,dataset_8, dataset_9, dataset_10, dataset_11,dataset_12])

test_ratio = 0.4
num_total = len(val_test_dataset)
num_test = int(test_ratio * num_total)
num_val = num_total - num_test
val_dataset, test_dataset = random_split(val_test_dataset, [num_val, num_test])



batch_size = 25
# Split the dataset into training, validation

# Create DataLoader for training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create DataLoader for test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f'-----------------------------------------------------------')
print(f'\n\nThe length of the training dataset is: {len(train_dataset)}')
print(f'-----------------------------------------------------------')



#----------------------------------------------------------------------------------------------------------#
#------------------------------------------Class Weights---------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

# Initialize a dictionary of counters to store the frequency of each label for each criterion
label_frequencies = defaultdict(lambda: defaultdict(int))

# Iterate over the dataset to count the frequency of each label for each criterion
for image, labels_dict in train_dataset:
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
total_samples = len(train_dataset)

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_4', NUM_EPOCHS = 100)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_4', NUM_EPOCHS = 100)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_4', NUM_EPOCHS = 100)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_4', NUM_EPOCHS = 100)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_4/Straightness of edges.pth'
}

output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_4"

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_3 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_3 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_3 - Integrity of edges', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_3 - Straightness of edges', NUM_EPOCHS = 100, INIT_LR = 1e-3)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_3/Straightness of edges.pth'
}


output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_3"

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_2 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_2 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_2 - Integrity of edges', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_LR1e_2 - Straightness of edges', NUM_EPOCHS = 100, INIT_LR = 1e-2)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_LR1e_2/Straightness of edges.pth'
}


output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_2"

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 100)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 100)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Integrity of edges', NUM_EPOCHS = 100)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_4 - Straightness of edges', NUM_EPOCHS = 100)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_4/Straightness of edges.pth'
}


output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_4_CW"

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Integrity of edges', NUM_EPOCHS = 100, INIT_LR = 1e-3)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_3 - Straightness of edges', NUM_EPOCHS = 100, INIT_LR = 1e-3)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_3/Straightness of edges.pth'
}


output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_3_CW"

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

trainer.train('Relative position and orientation between neighboring buildings', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Integrity of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Integrity of edges', NUM_EPOCHS = 100, INIT_LR = 1e-2)
trainer.train('Straightness of edges', 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2', 'New_Dataset/Plots Multiple Fine Tuned ResNet18_Class_LR1e_2 - Straightness of edges', NUM_EPOCHS = 100, INIT_LR = 1e-2)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

model_paths = {
    'Relative position and orientation between neighboring buildings': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'New_Dataset/Multiple Fine Tuned ResNet18_Class_LR1e_2/Straightness of edges.pth'
}


output_folder_test = "New_Dataset/Test_output/TEST_ResNet18_LR1e_2_CW"

evaluate_model(test_loader, model_paths, output_folder_test)
