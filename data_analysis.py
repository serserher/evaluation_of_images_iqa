##let's try to do some data analysis in the training images, first we need to get the data in a dataframe format
##we will need correlation matrices for the different parameters to see which ones actually matter
##we will need confusion matrices for testing
from evaluation_dataset import ImageDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from tqdm import tqdm

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/First100', 'shuffled_dataset/labels_First100.csv')
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', 'shuffled_dataset/labels_101to200.csv')
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/201to300', 'shuffled_dataset/labels_201to300.csv')
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv')
upsample_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings', 'upsampled/PositioningBuildings/upsampled_labels.csv')
upsample_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings1', 'upsampled/PositioningBuildings1/upsampled_labels.csv')
upsample_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings2', 'upsampled/PositioningBuildings2/upsampled_labels.csv')
upsample_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningBuildings3', 'upsampled/PositioningBuildings3/upsampled_labels.csv')
upsample_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads', 'upsampled/PositioningRoads/upsampled_labels.csv')
upsample_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads2', 'upsampled/PositioningRoads2/upsampled_labels.csv')
upsample_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads3', 'upsampled/PositioningRoads3/upsampled_labels.csv')
upsample_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/upsampled/PositioningRoads4', 'upsampled/PositioningRoads4/upsampled_labels.csv')
dataset_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/First25', 'shuffled_dataset_testing/labels_First25.csv')
dataset_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/26to50', 'shuffled_dataset_testing/labels_26to50.csv')
dataset_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/51to75', 'shuffled_dataset_testing/labels_51to75.csv')
dataset_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/76to100', 'shuffled_dataset_testing/labels_76to100.csv')


#dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8, dataset_5, dataset_6, dataset_7,dataset_8])
dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8])

#dataset = dataset_3
labels_list = []
#print(len(dataset))
for i in tqdm(range(len(dataset))):
    #print(i)
    aux_list = []
    label_value = 1.0 if dataset[i][1]["Relative position and orientation between neighboring buildings"].tolist() == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Position and orientation of buildings in relation to closest road/s"].tolist() == [1, 0] else 0.0
    aux_list.append(label_value)
    #label_value = 1.0 if dataset[i][1]["Building types in relation to underlying terrain type"].tolist() == [1, 0] else 0.0
    #aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Integrity of edges"].tolist() == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Straightness of edges"].tolist() == [1, 0] else 0.0
    aux_list.append(label_value)
    #label_value = 1.0 if dataset[i][1]["Size relative to type"].tolist() == [1, 0] else 0.0
    #aux_list.append(label_value)
    #label_value = 1.0 if dataset[i][1]["Conservation of color codi"].tolist() == [1, 0] else 0.0
    #aux_list.append(label_value)
    avg = 0
    for i in aux_list:
        avg += i/len(aux_list)
    aux_list.append(avg)
    #print(aux_list)
    labels_list.append(aux_list.copy())
#headers = ["Relative position and orientation between neighboring buildings", "Position and orientation of buildings in relation to closest road/s", "Building types in relation to underlying terrain type",
           #"Integrity of edges", "Straightness of edges", "Size relative to type", "Conservation of color codin", "Average rating of the image"]
headers = ["c1", "c2","c3","c4","avg"]

ratings_dataframe = pd.DataFrame(labels_list, columns = headers)
print(ratings_dataframe)

correlation_matrix = ratings_dataframe.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('./output plots/correlation_matrix_ALL_DATA.png')


# Calculate the average rating for each criterion
average_ratings = ratings_dataframe.mean()

# Fill the table with criterion and average ratings
table_content = ""
for criterion, rating in average_ratings.items():
    table_content += f"{criterion} & {rating:.2f} \\\\\n"

# Print the table
print("\\begin{table}[h]")
print("    \\centering")
print("    \\begin{tabular}{|c|c|}")
print("    \\hline")
print("    \\textbf{Criterion} & \\textbf{Average Rating} \\\\")
print("    \\hline")
print(table_content)
print("    \\hline")
print("    \\end{tabular}")
print("    \\caption{Average ratings for each criterion}")
print("    \\label{tab:average_ratings}")
print("\\end{table}")