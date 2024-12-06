"""
Component file used to handle all the image loading.
"""
# Inspiration for the dataloder taken from: https://github.com/arnoweng/CheXNet/blob/master/read_data.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataSet(Dataset):
    def __init__(self, labels_dict, amount, TrainingImages=None, ValidationImages=None, transform=None):
        # Hardcoded image directories
        self.image_dirs = [
            './Chest-X-rays14-Dataset/images_001/images/',
            './Chest-X-rays14-Dataset/images_002/images/',
            './Chest-X-rays14-Dataset/images_003/images/',
            './Chest-X-rays14-Dataset/images_004/images/',
            './Chest-X-rays14-Dataset/images_005/images/',
            './Chest-X-rays14-Dataset/images_006/images/',
            './Chest-X-rays14-Dataset/images_007/images/',
            './Chest-X-rays14-Dataset/images_008/images/',
            './Chest-X-rays14-Dataset/images_009/images/',
            './Chest-X-rays14-Dataset/images_010/images/',
            './Chest-X-rays14-Dataset/images_011/images/',
            './Chest-X-rays14-Dataset/images_012/images/',
        ]

        self.image_names = [] #list that will store the image names after reading from the file.
        self.labels_dict = labels_dict  # External labels dictionary
        self.transform = transform # Stores the image transformations to be applied (if any)
        if TrainingImages is not None:
            with open(TrainingImages, "r") as f: # opens the file specified by image_list_file for reading, and stores it in variable "f".
                for i, line in enumerate(f): #Code runs every line in the (f) file. I is the index. Line is the text. i.e First loop: i = 0, line = "apple"
                    if i >= amount:  # Stop after X lines
                        break
                    image_name = line.strip()  # Remove extra whitespace or newlines
                    self.image_names.append(image_name) # Appends each image name in the list to the "image_names" variable.

        if ValidationImages is not None:
            with open(ValidationImages, "r") as f: # opens the file specified by image_list_file for reading, and stores it in variable "f".
                for i, line in enumerate(f): #Code runs every line in the (f) file. I is the index. Line is the text. i.e First loop: i = 0, line = "apple"
                    if i >= amount:  # Stop after X lines
                        break
                    image_name = line.strip()  # Remove extra whitespace or newlines
                    self.image_names.append(image_name) # Appends each image name in the list to the "image_names" variable.

    def __getitem__(self, index):
        image_name = self.image_names[index]  # Get the image name at the given index
        for image_dir in self.image_dirs:  # Loops through all image directories, stores path in "image_dir"
            image_path = os.path.join(image_dir, image_name)  # Constructs the full path to the image
            # Full path is now stored as image_path i.e './Chest-X-rays14-Dataset/images_001/0001.png'
            if os.path.exists(image_path):  # Check if the image exists at the constructed path
                image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB format
                break  # Exit the loop once the image is found

        # Fetch the corresponding labels from the labels_dict using image_name (i.e 0001.png)
        label = self.labels_dict[image_name]

        # Apply the transform if it's provided
        if self.transform is not None:
            image = self.transform(image)

        return image,torch.FloatTensor(label) #Returns image (In rgb, and maybe transformed) and the labels as tensor.

    def __len__(self):
        return len(self.image_names)


# Standalone load_labels function (outside the class)
def load_labels(labels_file, CLASS_NAMES):
    all_diseases = CLASS_NAMES # Fetches List of all possible diseases from main code.

    # Create a mapping from disease name to index
    disease_to_idx = {disease: i for i, disease in enumerate(all_diseases)} # Makes map with amount like { 'Atelectasis': 0, 'Cardiomegaly': 1,

    labels_dict = {} #initializes an empty dictionary labels_dict, which will be used to store the binary label vectors for each image.

    # Read the labels file and process each line
    with open(labels_file, "r") as f: #opens the labels csv
        for line in f:
            items = line.split(',') #splits lines at , since its a csv
            image_name = items[0]  # First element is the image name/ID
            disease_labels = items[1]  # Second element contains the disease labels

            # Initialize a zero vector for the binary labels (size = len(all_diseases))
            label_vector = [0] * len(all_diseases) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] a 14 zero veector

            # Split the disease labels by '|', only proceed if there are disease labels
            if disease_labels and disease_labels != 'No Finding':
                diseases = disease_labels.split('|')
                # Set the corresponding index in the label vector to 1 for each disease
                for disease in diseases:
                    if disease in disease_to_idx:
                        label_vector[disease_to_idx[disease]] = 1

            # Store the label vector for this image
            labels_dict[image_name] = label_vector

    return labels_dict #returns a dictionary where each key is an image name and each value is the corresponding binary label vector.



# This function retrives the test images that are specified in self.image_names. These images ahve ground truth so we can compare CAM to ground truth.
class ChestXraySpecificImagesDataSet(Dataset):
    def __init__(self, transform=None):
        # Hardcoded image directories
        self.image_dirs = [
            './Chest-X-rays14-Dataset/images_001/images/',
            './Chest-X-rays14-Dataset/images_002/images/',
            './Chest-X-rays14-Dataset/images_003/images/',
            './Chest-X-rays14-Dataset/images_004/images/',
            './Chest-X-rays14-Dataset/images_005/images/',
            './Chest-X-rays14-Dataset/images_006/images/',
            './Chest-X-rays14-Dataset/images_007/images/',
            './Chest-X-rays14-Dataset/images_008/images/',
            './Chest-X-rays14-Dataset/images_009/images/',
            './Chest-X-rays14-Dataset/images_010/images/',
            './Chest-X-rays14-Dataset/images_011/images/',
            './Chest-X-rays14-Dataset/images_012/images/',
        ]

        # Hardcoded list of specific image IDs. You can replace it with any images here that have Ground truth. Check YOLOV5/runs/val for  results for that.
        self.image_names = [
            "00021860_003.png", "00022237_002.png", "00023026_008.png","00025221_001.png","00022899_014.png"
        ]

        self.transform = transform  # Image transformations to be applied (if any)

    def __getitem__(self, index):
        image_name = self.image_names[index]  # Get the image name at the given index

        # Loop through directories to find the image
        for image_dir in self.image_dirs:
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):  # Check if the image exists at the constructed path
                image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
                break  # Exit loop once the image is found

        # Apply transformations, if any
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)


