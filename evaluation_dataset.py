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
        self.__mapindexes__()
    def __mapindexes__(self):
        self.indexes_mappped = []
        for filename in os.listdir(self.path_to_images):
            if filename.endswith('.png'):
                image_number = int(filename.split('_')[-1].split('.')[0])
                self.indexes_mappped.append(image_number)
        self.indexes_mappped.sort()
    def __getitem__(self, index):
        working_index = self.indexes_mappped[index]
        
        # grab the image, label, and its bounding box coordinates
        image_path = os.path.join(self.path_to_images, '_' + f"{working_index}" + '.png')
        image = io.imread(image_path)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        if self.transforms:
            image = self.transforms(image)
        label_dic = {}
        i = 0
        while i<7:
            for header in self.headers:
                if ('_' + f"{working_index}" + '.png') in header:
                    label_index = self.headers.index(header)
                    header_name = header.strip('_' + f"{working_index}" + '.png')
                    if header_name not in label_dic:
                        if (self.criteriavalues[label_index]) == 1:
                            label_dic[header_name] = torch.tensor([1, 0])
                        else:
                            label_dic[header_name] = torch.tensor([0, 1])                            
                    i+=1

        # transpose the image such that its channel dimension becomes the leading one
        if self.transforms:
            image = self.transforms(image)
        # return a tuple of the images, labels, and bounding box coordinates
        return (image, label_dic)
    def __len__(self):
        # return the size of the dataset
        return int(len(self.criteriavalues)/7)

