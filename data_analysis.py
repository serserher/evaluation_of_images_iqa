##let's try to do some data analysis in the training images, first we need to get the data in a dataframe format
##we will need correlation matrices for the different parameters to see which ones actually matter
##we will need confusion matrices for testing
from evaluation_dataset import ImageDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from tqdm import tqdm

dataset_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/First75', 'New_Dataset/First75/labels_First75.csv')
dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/101to175', 'New_Dataset/101to175/labels_101to175.csv')
dataset_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/201to274', 'New_Dataset/201to274/labels_201to275.csv')
dataset_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/301to375', 'New_Dataset/301to375/labels_301to375.csv')
upsample_1 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings1/upsampled_labels.csv')
upsample_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings2/upsampled_labels.csv')
upsample_3 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings3', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings3/upsampled_labels.csv')
upsample_4 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings4', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Buildings4/upsampled_labels.csv')
upsample_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads1/upsampled_labels.csv')
upsample_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads2/upsampled_labels.csv')
upsample_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads3', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads3/upsampled_labels.csv')
upsample_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads4', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/Roads4/upsampled_labels.csv')
upsample_9 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges1', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges1/upsampled_labels.csv')
upsample_10 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges2', '/home/sergio/Thesis_Sergio/evaluation/New_Dataset/upsampled/StraightnessEdges2/upsampled_labels.csv')


#dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, upsample_1, upsample_2, upsample_3, upsample_4, upsample_5, upsample_6, upsample_7, upsample_8, upsample_9, upsample_10])

dataset_5 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/76to100', 'New_Dataset/76to100/labels_76to100.csv')
dataset_6 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/176to200', 'New_Dataset/176to200/labels_176to200.csv')
dataset_7 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/275to300', 'New_Dataset/275to300/labels_275to300.csv')
dataset_8 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/New_Dataset/376to400', 'New_Dataset/376to400/labels_376to400.csv')
dataset_9 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/First25', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_First25.csv')
dataset_10 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/26to50', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_26to50.csv')
dataset_11 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/51to75', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_51to75.csv')
dataset_12 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/76to100', '/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset_testing/labels_76to100.csv')

dataset = ConcatDataset([dataset_5, dataset_6, dataset_7,dataset_8, dataset_9, dataset_10, dataset_11,dataset_12])


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
plt.savefig('./output_plots/correlation_matrix_test_NewLessdata_upsampled.png')


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