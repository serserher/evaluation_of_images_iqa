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

Instructions for the architecture that trains the whole ResNets for each criterion will come soon.



