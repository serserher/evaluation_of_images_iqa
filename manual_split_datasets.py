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

path_to_images = '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400'
criteria_labels = 'shuffled_dataset/labels_301to400.csv'

output_path = 'New_Dataset/376to400'
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

count = 376
for image_index in range(376,401):
    labels_dic = {}
    path_to_image = f'{path_to_images}/_{image_index}.png'
    try:
        image = io.imread(path_to_image)
    except:
        continue
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
filename_labels = 'New_Dataset/376to400/labels_376to400.csv'
# Open the file in write mode


with open(filename_labels, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(new_headers)
    
    # Write the values
    writer.writerow(new_criteriavalues)

print(f"CSV file '{filename_labels}' has been generated successfully.")