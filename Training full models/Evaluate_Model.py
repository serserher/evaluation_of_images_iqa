from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix as C_Matrix
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(test_loader, model_paths):
    criteria_names = ["Relative position and orientation between neighboring buildings", 
                      "Position and orientation of buildings in relation to closest road/s", 
                      "Integrity of edges", "Straightness of edges"]

    # Initialize dictionaries to store evaluation results for each criterion
    confusion_matrices_dict = {criterion: [] for criterion in criteria_names}
    f1_scores_dict = {criterion: [] for criterion in criteria_names}
    precisions_dict = {criterion: [] for criterion in criteria_names}
    recalls_dict = {criterion: [] for criterion in criteria_names}
    accuracies_dict = {criterion: [] for criterion in criteria_names}

    # Loop through each criterion
    for criterion, model_path in model_paths.items():
        # Load the model
        perf_evaluator_model = torch.load(model_path)
        perf_evaluator_model.eval()  # Set model to evaluation mode
        
        # Loop through the test dataset
        for (images, criteria) in tqdm(test_loader):
            images = images.to(DEVICE)

            # Perform forward pass and get predictions
            with torch.no_grad():
                predictions = perf_evaluator_model(images)
            
            # Assuming predictions are probabilities, convert to binary predictions
            pred_labels = 1 - predictions.float().cpu().numpy()
            true_labels = 1 - criteria[criterion].float().cpu().numpy()
            single_label_pred = np.argmax(pred_labels, axis=1)
            single_label_true = np.argmax(true_labels, axis=1)

            # Calculate evaluation metrics
            confusion_matrix_single = C_Matrix(single_label_true, single_label_pred)
            f1_score_single = f1_score(single_label_true, single_label_pred)
            precision_single = precision_score(single_label_true, single_label_pred)
            recall_single = recall_score(single_label_true, single_label_pred)
            accuracy_single = accuracy_score(single_label_true, single_label_pred)
            
            # Store evaluation results
            confusion_matrices_dict[criterion].append(confusion_matrix_single)
            f1_scores_dict[criterion].append(f1_score_single)
            precisions_dict[criterion].append(precision_single)
            recalls_dict[criterion].append(recall_single)
            accuracies_dict[criterion].append(accuracy_single)

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

    # Plot and save confusion matrices
    experiment_id = "1_ResNet18_for_each_criterion"  # Replace this with your unique identifier
    output_folder = os.path.join("output_plots", experiment_id)
    os.makedirs(output_folder, exist_ok=True)
    for criterion_name in criteria_names:
        # Retrieve the confusion matrix for the current criterion
        confusion_matrix = aggregate_results[criterion_name]["confusion_matrix"]

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
            filename = os.path.join(output_folder, "Position_and_orientation_of_buildings_in_relation_to_closest_roads_confusion_matrix.png")
            plt.savefig(filename)
            plt.close()
        else:
            # Save the plot with the criterion name as the filename
            filename = os.path.join(output_folder, f"{criterion_name}_confusion_matrix.png")
            plt.savefig(filename)
            plt.close()

        print(f"Confusion matrix plot saved for criterion: {criterion_name}")
