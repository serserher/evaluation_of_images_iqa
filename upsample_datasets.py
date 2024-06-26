from evaluation_dataset import ImageDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import os
import re
import copy
from skimage import io
import csv

"""
dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/First75', 'New_Dataset/First75/labels_First75.csv')
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/101to175', 'New_Dataset/101to175/labels_101to175.csv')
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/201to274', 'New_Dataset/201to274/labels_201to275.csv')
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/301to375', 'New_Dataset/301to375/labels_301to375.csv')
"""

path_to_images = '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/101to175'
criteria_labels = 'New_Dataset/101to175/labels_101to175.csv'

output_path = 'New_Dataset/upsampled/StraightnessEdges2'
# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
criteria_labels = pd.read_csv(criteria_labels)
headers = criteria_labels.columns.tolist()
headers.pop(0)
criteriavalues_lists = criteria_labels.values.tolist()
criteriavalues_lists[0].pop(0)
criteriavalues = criteriavalues_lists[0]
new_criteriavalues = ["values"]
new_headers = ["headers"]

indexes_mappped = []
for filename in os.listdir(path_to_images):
    if filename.endswith('.png'):
        image_number = int(filename.split('_')[-1].split('.')[0])
        indexes_mappped.append(image_number)
    indexes_mappped.sort()

#For simplicity, here we can just initialize count as the last element from the previous dataset that we are about to continue with more upsampling    
#count = copy.deepcopy(indexes_mappped[-1]) + 1
count = 827 + 1
criterion_to_upsample = "Straightness of edges"
value_to_upsample = 0
upsample_this_times = 4
images_to_upsample = []
for header in headers:
    if criterion_to_upsample in header:
        index = headers.index(header)
        if criteriavalues[index] == value_to_upsample:
            image_number = re.search(r'_(\d+).png', header).group(1)
            images_to_upsample.append(image_number)
            
for image_index in images_to_upsample:
    labels_dic = {}
    path_to_image = f'{path_to_images}/_{image_index}.png'
    image = io.imread(path_to_image)
    for _ in range(upsample_this_times):
        for header in headers:
            if ('_' + f"{image_index}" + '.png') in header:
                modified_header = header.replace(f"_{image_index}.png", f"_{count}.png")
                new_headers.append(modified_header)
                index = headers.index(header)
                new_criteriavalues.append(criteriavalues[index])
        img_name = f'_{count}.png'
        save_dir = os.path.join(output_path, img_name)
        #print(save_dir)
        io.imsave(save_dir, image)
        count += 1
                    
#print(new_headers, new_criteriavalues, count)
filename_labels = 'New_Dataset/upsampled/StraightnessEdges2/upsampled_labels.csv'
# Open the file in write mode


with open(filename_labels, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(new_headers)
    
    # Write the values
    writer.writerow(new_criteriavalues)

print(f"CSV file '{filename_labels}' has been generated successfully.")