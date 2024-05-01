#code to reshuffle datasets and add appropiate naming to the files for them to be processed by the rest of the scripts
import os
from skimage import io
import random
from tqdm import tqdm

input_path = "/home/sergio/Thesis_Sergio/inference_repo/blockgen_inference/outputs/reshuffle_testing_dataset"
output_path = "./shuffled_dataset_testing"

filenames = [filename for filename in os.listdir(input_path) if filename.endswith(".png")]
# Shuffle the list of filenames, this will randomize the order they appear in
random.shuffle(filenames)

if not os.path.isdir(output_path):
    os.makedirs(output_path)
index = 1
for filename in tqdm(filenames):
    if filename.endswith(".png"):
        image = io.imread(os.path.join(input_path, filename))
        save_dir = os.path.join(output_path, f"_{index}.png")
        io.imsave(save_dir, image)
        index += 1