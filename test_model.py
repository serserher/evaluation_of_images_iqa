from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from evaluation_dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
target_size = (768, 768)

transforms_regular = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

TestDataset = ImageDataset('/home/sergio/Thesis_Sergio/evaluation/shuffled_dataset/101to200', 'shuffled_dataset/labels_101to200.csv', transforms=transforms_regular)

criteria_names = ["Relative position and orientation between neighboring buildings", "Position and orientation of buildings in relation to closest road/s", "Building types in relation to underlying terrain type", 
                            "Integrity of edges", "Straightness of edges", "Size relative to type", "Conservation of color codi"]
# Initialize dictionaries to store evaluation results for each criterion
confusion_matrices_dict = {criterion: [] for criterion in criteria_names}
f1_scores_dict = {criterion: [] for criterion in criteria_names}
precisions_dict = {criterion: [] for criterion in criteria_names}
recalls_dict = {criterion: [] for criterion in criteria_names}
accuracies_dict = {criterion: [] for criterion in criteria_names}
test_loader = torch.utils.data.DataLoader(TestDataset, batch_size=50, shuffle = True)

perf_evaluator_model = torch.load('output model/performance_evaluator.pth')

# Loop through the batches in the test loader
for (images, criteria) in tqdm(test_loader):
    images = images.to(DEVICE)
    predictions = perf_evaluator_model(images)
    predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions]

    # Loop through each criterion and calculate evaluation metrics
    for i in range(len(predictions_squeezed)):
        criterion_name = criteria_names[i]  # Assuming the criteria are numbered from 1 to 7
        # Calculate confusion matrix
        pred_labels = (predictions_squeezed[i] >= 0.5).cpu().numpy().flatten()
        true_labels = criteria[criterion_name].cpu().numpy().flatten()
        if any(math.isnan(x) for x in true_labels):
            true_labels = np.nan_to_num(true_labels, nan=0)
        confusion_matrices_dict[criterion_name].append(confusion_matrix(true_labels, pred_labels))

        # Calculate F1 score
        f1_scores_dict[criterion_name].append(f1_score(true_labels, pred_labels, average='macro'))

        # Calculate precision
        precisions_dict[criterion_name].append(precision_score(true_labels, pred_labels, average='macro'))

        # Calculate recall
        recalls_dict[criterion_name].append(recall_score(true_labels, pred_labels, average='macro'))
        
        # Calculate accuracy
        accuracies_dict[criterion_name].append(accuracy_score(true_labels, pred_labels))

# Aggregate the evaluation results for each criterion
aggregate_results = {}
num_batches = len(test_loader)
for criterion in criteria_names:
    aggregate_results[criterion] = {
        "confusion_matrix": np.sum(confusion_matrices_dict[criterion], axis=0),
        "f1_score": np.sum(f1_scores_dict[criterion]) / num_batches,
        "precision": np.sum(precisions_dict[criterion]) / num_batches,
        "recall": np.sum(recalls_dict[criterion]) / num_batches,
        "accuracy": np.sum(accuracies_dict[criterion]) / num_batches
    }

# Print the aggregated results
for criterion, results in aggregate_results.items():
    print("Criterion:", criterion)
    print("Total Confusion Matrix:")
    print(results["confusion_matrix"])
    print("Total F1 Score:", results["f1_score"])
    print("Total Precision:", results["precision"])
    print("Total Recall:", results["recall"])
    print("Total Accuracy:", results["accuracy"])
    

# Calculate overall average F1 score, precision, recall, and accuracy
overall_f1_score = np.mean([np.mean(f1_scores_dict[criterion]) for criterion in criteria_names])
overall_precision = np.mean([np.mean(precisions_dict[criterion]) for criterion in criteria_names])
overall_recall = np.mean([np.mean(recalls_dict[criterion]) for criterion in criteria_names])
overall_accuracy = np.mean([np.mean(accuracies_dict[criterion]) for criterion in criteria_names])

print("Overall Average F1 Score:", overall_f1_score)
print("Overall Average Precision:", overall_precision)
print("Overall Average Recall:", overall_recall)
print("Overall Average Accuracy:", overall_accuracy)


experiment_id = "testing the test"  # Replace this with your unique identifier
output_folder = os.path.join("output plots", experiment_id)
os.makedirs(output_folder, exist_ok=True)
criteria_names_directory = ["Relative position and orientation between neighboring buildings", "Position and orientation of buildings in relation to closest roads", "Building types in relation to underlying terrain type", 
                            "Integrity of edges", "Straightness of edges", "Size relative to type", "Conservation of color codin"]
for criterion_name in criteria_names:
    # Retrieve the confusion matrix for the current criterion
    confusion_matrix = aggregate_results[criterion_name]["confusion_matrix"]
    #confusion_matrix = np.array(confusion_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size for better readability
    ConfusionMatrixDisplay(confusion_matrix).plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {criterion_name}", fontsize=10)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    if criterion_name == "Position and orientation of buildings in relation to closest road/s":
        # Save the plot with the criterion name as the filename
        filename = os.path.join(output_folder, "Position and orientation of buildings in relation to closest roads_confusion_matrix.png")
        plt.savefig(filename)
        plt.close()
    else:
        # Save the plot with the criterion name as the filename
        filename = os.path.join(output_folder, f"{criterion_name}_confusion_matrix.png")
        plt.savefig(filename)
        plt.close()

    print(f"Confusion matrix plot saved for criterion: {criterion_name}")