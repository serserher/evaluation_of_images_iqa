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
from torch.utils.data import ConcatDataset


class TestPipeline:
    def __init__(self, test_loader, trained_model):
        self.test_loader = test_loader
        self.trained_model = trained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criteria_names = ["Relative position and orientation between neighboring buildings",
                               "Position and orientation of buildings in relation to closest road/s",
                               "Integrity of edges", "Straightness of edges"]
        self.confusion_matrices_dict = {criterion: [] for criterion in self.criteria_names}
        self.f1_scores_dict = {criterion: [] for criterion in self.criteria_names}
        self.precisions_dict = {criterion: [] for criterion in self.criteria_names}
        self.recalls_dict = {criterion: [] for criterion in self.criteria_names}
        self.accuracies_dict = {criterion: [] for criterion in self.criteria_names}

    def evaluate(self):
        self.trained_model.eval()
        with torch.no_grad():
            for images, criteria in self.test_loader:
                images = images.to(self.device)
                predictions = self.trained_model(images)
                predictions_squeezed = [torch.squeeze(pred, dim=1) for pred in predictions]

                for i in range(len(predictions_squeezed)):
                    criterion_name = self.criteria_names[i]
                    pred_labels = (predictions_squeezed[i] > 0.5).float().cpu().numpy()
                    true_labels = criteria[criterion_name].float().cpu().numpy()

                    pred_labels = 1 - pred_labels
                    true_labels = 1 - true_labels
                    single_label_pred = np.argmax(pred_labels, axis=1)
                    single_label_true = np.argmax(true_labels, axis=1)

                    true_labels = np.nan_to_num(true_labels, nan=0)
                    self.confusion_matrices_dict[criterion_name].append(confusion_matrix(single_label_true, single_label_pred))
                    self.f1_scores_dict[criterion_name].append(f1_score(true_labels, pred_labels, average='macro'))
                    self.precisions_dict[criterion_name].append(precision_score(true_labels, pred_labels, average='macro'))
                    self.recalls_dict[criterion_name].append(recall_score(true_labels, pred_labels, average='macro'))
                    self.accuracies_dict[criterion_name].append(accuracy_score(true_labels, pred_labels))

    def aggregate_results(self):
        aggregate_results = {}
        num_batches = len(self.test_loader)
        for criterion in self.criteria_names:
            aggregate_results[criterion] = {
                "confusion_matrix": np.sum(self.confusion_matrices_dict[criterion], axis=0),
                "f1_score": np.sum(self.f1_scores_dict[criterion]) / num_batches,
                "precision": np.sum(self.precisions_dict[criterion]) / num_batches,
                "recall": np.sum(self.recalls_dict[criterion]) / num_batches,
                "accuracy": np.sum(self.accuracies_dict[criterion]) / num_batches
            }
        return aggregate_results
    
    def print_aggregated_results(self, output_folder):
        aggregated_results = self.aggregate_results()
        output_file = os.path.join(output_folder, "experiment_results.txt")
        with open(output_file, "w") as f:
            for criterion, results in aggregated_results.items():
                f.write(f"Criterion: {criterion}\n")
                f.write("Total Confusion Matrix:\n")
                f.write(str(results["confusion_matrix"]) + "\n")
                f.write(f"Total F1 Score: {results['f1_score']}\n")
                f.write(f"Total Precision: {results['precision']}\n")
                f.write(f"Total Recall: {results['recall']}\n")
                f.write(f"Total Accuracy: {results['accuracy']}\n\n")

            overall_f1_score = np.mean([np.mean(self.f1_scores_dict[criterion]) for criterion in self.criteria_names])
            overall_precision = np.mean([np.mean(self.precisions_dict[criterion]) for criterion in self.criteria_names])
            overall_recall = np.mean([np.mean(self.recalls_dict[criterion]) for criterion in self.criteria_names])
            overall_accuracy = np.mean([np.mean(self.accuracies_dict[criterion]) for criterion in self.criteria_names])

            f.write(f"Overall Average F1 Score: {overall_f1_score}\n")
            f.write(f"Overall Average Precision: {overall_precision}\n")
            f.write(f"Overall Average Recall: {overall_recall}\n")
            f.write(f"Overall Average Accuracy: {overall_accuracy}\n")

    def plot_confusion_matrices(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for criterion_name in self.criteria_names:
            confusion_matrix = self.aggregate_results()[criterion_name]["confusion_matrix"]
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            ConfusionMatrixDisplay(confusion_matrix).plot(cmap='Blues')
            plt.title(f"Confusion Matrix - {criterion_name}", fontsize=10)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()
            if criterion_name == "Position and orientation of buildings in relation to closest road/s":
                filename = os.path.join(output_folder, "Position and orientation of buildings in relation to closest roads_confusion_matrix.png")
            else:
                filename = os.path.join(output_folder, f"{criterion_name}_confusion_matrix.png")
            plt.savefig(filename)
            plt.close()
            print(f"Confusion matrix plot saved for criterion: {criterion_name}")
