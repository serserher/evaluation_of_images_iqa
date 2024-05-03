##let's try to do some data analysis in the training images, first we need to get the data in a dataframe format
##we will need correlation matrices for the different parameters to see which ones actually matter
##we will need confusion matrices for testing
from evaluation_dataset import ImageDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

#dataset = ImageDataset ('/home/sergio/Thesis_Sergio/inference/output/inference', 'labels.csv')
upsample_config = [("Conservation of color coding", 0)]

dataset = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv',  upsample_list=upsample_config, upsample_factor = 3)
#dataset_2 = ImageDataset ('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/301to400', 'shuffled_dataset/labels_301to400.csv')
#dataset = ConcatDataset([dataset_1, dataset_2])
dataset.upsample()

labels_list = []
print(len(dataset))
for i in range(len(dataset)):
    print(i)
    aux_list = []
    label_value = 1.0 if dataset[i][1]["Relative position and orientation between neighboring buildings"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Position and orientation of buildings in relation to closest road/s"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Building types in relation to underlying terrain type"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Integrity of edges"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Straightness of edges"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Size relative to type"] == [1, 0] else 0.0
    aux_list.append(label_value)
    label_value = 1.0 if dataset[i][1]["Conservation of color codi"] == [1, 0] else 0.0
    aux_list.append(label_value)
    avg = 0
    for i in aux_list:
        avg += i/len(aux_list)
    aux_list.append(avg)
    labels_list.append(aux_list.copy())
#headers = ["Relative position and orientation between neighboring buildings", "Position and orientation of buildings in relation to closest road/s", "Building types in relation to underlying terrain type",
           #"Integrity of edges", "Straightness of edges", "Size relative to type", "Conservation of color codin", "Average rating of the image"]
headers = ["c1", "c2", "c3","c4","c5","c6","c7","avg"]

ratings_dataframe = pd.DataFrame(labels_list, columns = headers)
print(ratings_dataframe)

correlation_matrix = ratings_dataframe.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('./output plots/upsampled/correlation_matrix_dataset_eval_Eric.png')


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