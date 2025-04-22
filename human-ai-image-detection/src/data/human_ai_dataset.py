import os
import random
from typing import Dict, List
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import env as env
from PIL import Image
import kagglehub
import shutil
import os

# Reference article: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class HumanVSAIDataset(Dataset):
    """Dataset model for loading the Human VS AI dataset as an object"""
    def __init__(self, annotations_file_path: str, transform=None, target_transform=None, split = 'train'):
        self.img_dir = env.DATASET_DIR
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = pd.read_csv(annotations_file_path)
        target_path = env.DATASET_DIR
        if os.path.exists(target_path):
            print(f"Dataset already exists at: {target_path}")
        else:
            self.load_dataset()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)

        if image.shape[0] == 1:  # Grayscale images have 1 channel
            image = image.repeat(3, 1, 1)  # Repeat the single channel across RGB

        image = image.float()  # Convert to float32
        image /= 255.0  # Scale to [0, 1]

        # NOTE: annotations file format - ID, image path, label (0,1)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def load_dataset(self):
        print("Downloading dataset...")
        download_path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
        print("Downloaded to:", download_path)

        target_path = os.path.join("dataset")

        # Step 3: Create the directory if it doesn't exist
        os.makedirs(target_path, exist_ok=True)

        for item in os.listdir(download_path):
            source = os.path.join(download_path, item)
            destination = os.path.join(target_path, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)

        print(f"Dataset moved to: {target_path}")


    def get_dataset(self, subset_size = None) -> Dict[str, List]:
        """Parses and returns the dataset as a `Dict`"""
        labels_map = {0: 'Human Generated', 1: 'AI Generated'}
        
        dataset: Dict[str, List] = {
            'ids': [],
            'images': [],
            'labels': [],
            'label_names': [],
        }
        
        for i in range(len(self.img_labels)) if not subset_size else random.sample(range(len(self.img_labels)), subset_size):
            img_id = self.img_labels.iloc[i, 0]
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 1])
            image_tensor = read_image(img_path)

            # Convert grayscale to RGB by stacking if needed
            if image_tensor.shape[0] == 1:  # Grayscale images have 1 channel
                image_tensor = image_tensor.repeat(3, 1, 1)  # Repeat the single channel across RGB
                print('detected greyscale image')

            # NOTE: annotations file format - ID, image path, label (0,1)
            label = self.img_labels.iloc[i, 2]

            dataset['ids'].append(img_id)
            if self.transform:
                image_tensor = self.transform(image_tensor)

            dataset['images'].append(image_tensor)
            dataset['label_names'].append(labels_map[label])

            if self.target_transform:
                label = self.transform(label)
            dataset['labels'].append(label)    

        return dataset