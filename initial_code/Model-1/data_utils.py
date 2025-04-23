import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class ImageLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.csv_dir = os.path.dirname(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.csv_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 2]  # Assuming label is in the third column
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(csv_file, batch_size=32, train_split=0.8, transform=None, num_images=None):
    """
    Splits the dataset into training and testing sets and returns data loaders.

    Parameters:
    csv_file (str): Path to the CSV file containing image paths and labels.
    batch_size (int): Number of samples per batch.
    train_split (float): Proportion of the dataset to include in the train split.
    transform (callable, optional): A function/transform to apply to the images.

    Returns:
    train_loader, test_loader: Data loaders for the training and testing sets.
    """
    dataset = ImageLabelDataset(csv_file, transform=transform)
    if num_images is not None:
        dataset.data_frame = dataset.data_frame.head(num_images)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
transform = transforms.Compose([
    transforms.Resize((768, 512)),
    transforms.ToTensor()
])

#train_loader, test_loader = get_data_loaders('path/to/train.csv', batch_size=32, transform=transform) 