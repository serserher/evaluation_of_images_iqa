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

validation_ratio = 0.2
test_ratio = 0.1
num_total = len(dataset)
num_valid = int(validation_ratio * num_total)
num_test = int(test_ratio * num_total)
num_train = num_total - num_valid - num_test

batch_size = 15
# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_valid, num_test])

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

trainer = CustomTrainer(train_loader, val_loader, class_weights)

trainer.train('Relative position and orientation between neighboring buildings', 'Multiple Fine Tuned ResNet18_Class', 'Plots Multiple Fine Tuned ResNet18_Class - Relative position and orientation between neighboring buildings', NUM_EPOCHS = 10)
trainer.train('Position and orientation of buildings in relation to closest road/s', 'Multiple Fine Tuned ResNet18_Class', 'Plots Multiple Fine Tuned ResNet18_Class - Position and orientation of buildings in relation to closest roads', NUM_EPOCHS = 10)
trainer.train('Integrity of edges', 'Multiple Fine Tuned ResNet18_Class', 'Plots Multiple Fine Tuned ResNet18_Class - Integrity of edges', NUM_EPOCHS = 10)
trainer.train('Straightness of edges', 'Multiple Fine Tuned ResNet18_Class', 'Plots Multiple Fine Tuned ResNet18_Class - Straightness of edges', NUM_EPOCHS = 10)

model_paths = {
    'Relative position and orientation between neighboring buildings': 'Multiple Fine Tuned ResNet18_Class/Integrity of edges.pth',
    'Position and orientation of buildings in relation to closest road/s': 'Multiple Fine Tuned ResNet18_Class/Position and orientation of buildings in relation to closest roads.pth',
    'Integrity of edges': 'Multiple Fine Tuned ResNet18_Class/Relative position and orientation between neighboring buildings.pth',
    'Straightness of edges': 'Multiple Fine Tuned ResNet18_Class/Straightness of edges.pth'
}

evaluate_model(test_loader, model_paths)







