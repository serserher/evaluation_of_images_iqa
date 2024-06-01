# evaluation_of_images_iqa
A repository for an image quality analysis CNN implementation to evaluate the quality of images generated with generative AI models.


# Training and Testing Pipeline with shared backbone and multiple convolutional classifying heads.

This repository contains scripts for training and testing a shared backbone architecture with multiple convolutional classifying heads.

## Training

The training process begins with the backbone architecture defined in `performance_evaluator.py`, which is called by the `TrainingPipeline` class in `Training_Pipeline.py`. To utilize this method, follow these steps:

1. Import the `TrainingPipeline` class into your script:

    ```python
    from Training_Pipeline import TrainingPipeline
    ```

2. Instantiate the `TrainingPipeline` class with the appropriate parameters, such as train and validation datasets, backbone model, learning rate, batch size, and number of epochs:

    ```python
    pipeline = TrainingPipeline(train_dataset, val_dataset, backbone_model, init_lr=1e-4, batch_size=25, num_epochs=100)
    ```

3. Define your train and validation datasets using the `ImageDataset` class. For example:

    ```python
    dataset_1 = ImageDataset('/path/to/train_data', '/path/to/labels.csv', transforms=transforms_regular)
    ```
**Disclaimer about the data:** This project is part of a Master's Thesis in collaboration with a company so the data used will be kept private.

- The format for the train data includes:
  - **Images**
  - **Labels:** A CSV file with binary ratings for each image

- The format of the rating headers is as follows (example with 100 images and 2 criteria):
  - The format of the rating values is: [values, 1, 0, 1, 1, ..., 1, 0, 1, 0, 0, 0, 0]

  - The format of the rating headers is: [headers, _1.png(name_of_criterion_1), _1.png(name_of_criterion_2), _2.png(name_of_criterion_1), _2.png(name_of_criterion_2), ..., _100.png(name_of_criterion_1), _100.png(name_of_criterion_2)]
    
4. Initiate the training process by calling the `train()` function:

    ```python
    pipeline.train()
    ```

5. After training, save the trained models and metric plots using the respective object functions:

    ```python
    pipeline.save_models("directory_to_save_models")
    pipeline.plot_metrics("directory_to_save_plots")
    ```

## Testing

To assess the trained model, you can use the `TestPipeline` class defined in `Testing_Pipeline.py`. Follow these steps:

1. Import the `TestPipeline` class into your script:

    ```python
    from Testing_Pipeline import TestPipeline
    ```

2. Instantiate the `TestPipeline` class with the testing dataset loader and the trained model:

    ```python
    test_pipeline = TestPipeline(test_loader, perf_evaluator_model)
    ```

3. Execute the test by calling the `evaluate()` function:

    ```python
    test_pipeline.evaluate()
    ```

4. Visualize and store the test results using the following set of functions:

    ```python
    aggregate_results = test_pipeline.aggregate_results()
    output_folder = "output_directory"
    test_pipeline.plot_confusion_matrices(output_folder)
    test_pipeline.print_aggregated_results(output_folder)
    ```

## Example Usage

For a complete example of how to use the training and testing pipelines, refer to Train_And_Test.py file.

# Training and Testing Pipeline with full models per criterion.


1. Import the `CustomTrainer` class into your script:

    ```python
    from Custom_Trainer import CustomTrainer
    ```

    
2. Define your train and validation datasets using the `ImageDataset` class. For example:

    ```python
    dataset_1 = ImageDataset('/path/to/train_data', '/path/to/labels.csv', transforms=transforms_regular)
    ```
    Instantiate the DataLoaders as normally done in these kinds of Pytorch implementations, an example can be found here:
   ```python
   from torch.utils.data import DataLoader
   # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create DataLoader for validation set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   ```
    
**Disclaimer about the data:** This project is part of a Master's Thesis in collaboration with a company so the data used will be kept private.

- The format for the train data includes:
  - **Images**
  - **Labels:** A CSV file with binary ratings for each image

- The format of the rating headers is as follows (example with 100 images and 2 criteria):
  - The format of the rating values is: [values, 1, 0, 1, 1, ..., 1, 0, 1, 0, 0, 0, 0]

  - The format of the rating headers is: [headers, _1.png(name_of_criterion_1), _1.png(name_of_criterion_2), _2.png(name_of_criterion_1), _2.png(name_of_criterion_2), ..., _100.png(name_of_criterion_1), _100.png(name_of_criterion_2)]

  
3. Instantiate the `CustomTrainer` class with the appropriate parameters, such as train and validation loaders, differently to the previously described multihead approach you have to first instantiate the loaders and that is what you pass as an argument instead of the datasets:

    ```python
    trainer = CustomTrainer(train_loader, val_loader)
    ```

4. Once per criterion, initiate the training process by calling the `train()` function from the class:
    trainer.train('name of the criterion', 'model_path (where you want to save the model)', 'figures_path (where you want to save the figures)', NUM_EPOCHS, INIT_LR)

## Testing

To assess the trained model, you can use the `TestPipeline` class defined in `Testing_Pipeline.py`. Follow these steps:

1. Import the `evaluate_model` function into your script:

    ```python
    from Evaluate_Model import evaluate_model
    ```

2. Once per criterion, call the function passing as arguments the test dataloader, the model path where you stored the trained model for the specific criterion, and the output folder where you would like to store the metrics:
   
   ```python
   evaluate_model(test_loader, model_paths, output_folder_test)
   ```
## Example Usage

For a complete example of how to use the training and testing pipelines, refer to Train_Test_Models.py file in the "Training full models" folder.




