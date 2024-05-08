from torch.nn import Sequential
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.models import resnet50, resnet18, ResNet18_Weights, mobilenet_v2
from torch.utils.data import DataLoader
from evaluation_dataset import ImageDataset
from Training_Pipeline import TrainingPipeline
from Testing_Pipeline import TestPipeline
from collections import defaultdict
import torch
from torch.utils.data.dataset import random_split


base_model = resnet18(pretrained=True)
"""for param in base_model.parameters():
	param.requires_grad = False
"""
# print(base_model)
# Truncate the backbone model after the desired layer (layer4)
backbone_model = Sequential(
    base_model.conv1,
    base_model.bn1,
    base_model.relu,
    base_model.maxpool,
    base_model.layer1,
    base_model.layer2,
    base_model.layer3,
    base_model.layer4,
    )

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
dataset_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/First25', 'shuffled_dataset_testing/labels_First25.csv')
dataset_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/26to50', 'shuffled_dataset_testing/labels_26to50.csv')
dataset_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/51to75', 'shuffled_dataset_testing/labels_51to75.csv')
dataset_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/76to100', 'shuffled_dataset_testing/labels_76to100.csv')

dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8])


print(f'---------------------------------------------------')
print(f'\n\nThe length of the dataset is: {len(dataset)}')
print(f'---------------------------------------------------')


test_ratio = 0.2
num_total = len(dataset)
num_test = int(test_ratio * num_total)
num_train = num_total - num_test
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])


#----------------------------------------------------------------------------------------------------------#
#------------------------------------------Class Weights---------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

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

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-4, batch_size = 25, num_epochs=50)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_4")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_4_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_4/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_4"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)

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

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-3, batch_size = 25, num_epochs=50)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_3")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_3_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_3/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_3"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)

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

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-2, batch_size = 25, num_epochs=50)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_2")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_2_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_2/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_2"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#--------------------------------------LR1e-4 Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-4, batch_size = 25, num_epochs=50, class_weights = class_weights)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_4_WC")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_4_WC_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_4_WC/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_4_WC"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#--------------------------------------LR1e-3 Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-3, batch_size = 25, num_epochs=50, class_weights = class_weights)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_3_WC")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_3_WC_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_3_WC/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_3_WC"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#---------------------------------------------ResNet18-----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#--------------------------------------LR1e-2 Weighted classes---------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------Train-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

pipeline = TrainingPipeline(train_dataset, backbone_model, init_lr = 1e-2, batch_size = 25, num_epochs=50, class_weights = class_weights)
pipeline.train()
pipeline.save_models("ResNet18_Conv_MH_1e_2_WC")
pipeline.plot_metrics("ResNet18_Conv_MH_1e_2_WC_Training_Metrics")

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------TEST--------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

test_loader = DataLoader(test_dataset, batch_size=25, shuffle = True)
perf_evaluator_model = torch.load('ResNet18_Conv_MH_1e_2_WC/full_model.pth')

test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
test_pipeline.evaluate()
aggregate_results = test_pipeline.aggregate_results()
output_folder = "output_plots/ResNet18_Conv_MH_1e_2_WC"
test_pipeline.plot_confusion_matrices(output_folder)
test_pipeline.print_aggregated_results(output_folder)