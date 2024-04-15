from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
import torch 

#labels_csv = pd.read_csv('labels.csv')
# import the necessary packages
class ImageDataset(Dataset):
	# initialize the constructor
    def __init__(self, path_to_images, criteria_labels, transforms=None):
        self.path_to_images = path_to_images
        self.criteria_labels = pd.read_csv(criteria_labels)
        self.transforms = transforms
        self.headers = self.criteria_labels.columns.tolist()
        self.headers.pop(0)
        self.criteriavalues_lists = self.criteria_labels.values.tolist()
        self.criteriavalues_lists[0].pop(0)
        self.criteriavalues = self.criteriavalues_lists[0]
        
    def __getitem__(self, index):
        index = index + 1
        # grab the image, label, and its bounding box coordinates
        image_path = os.path.join(self.path_to_images, '_' + f"{index}" + '.jpg')
        image = io.imread(image_path)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        if self.transforms:
            image = self.transforms(image)
        label_dic = {}
        i = 0
        while i<7:
            for header in self.headers:
                if ('_' + f"{index}" + '.jpg') in header:
                    label_index = self.headers.index(header)
                    header_name = header.strip('_' + f"{index}" + '.jpg')
                    if header_name not in label_dic:
                        label_dic[header_name] = float(self.criteriavalues[label_index])
                    i+=1

        # transpose the image such that its channel dimension becomes the leading one
        if self.transforms:
            image = self.transforms(image)
        # return a tuple of the images, labels, and bounding box coordinates
        return (image, label_dic)
    def __len__(self):
        # return the size of the dataset
        return int(len(self.criteriavalues)/7)
"""      
dataset = ImageDataset ('/home/sergio/Thesis_Sergio/inference/output/inference', 'labels.csv')
[image, labels] = dataset.__getitem__(5)
io.imsave(f'image_2.jpg', image)
print(dataset.__len__())
"""