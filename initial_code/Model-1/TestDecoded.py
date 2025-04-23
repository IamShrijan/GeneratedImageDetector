import torch
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from Autoencoder import CNNAutoencoderWithDropout

# Create an instance of the model
autoencoder = CNNAutoencoderWithDropout()

# Load the model weights
state_dict = torch.load('autoencoder_epoch_1.pth')
autoencoder.load_state_dict(state_dict)
autoencoder.eval()  # Set the model to evaluation mode

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((768, 512)),  # Resize to match the input size
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to display the original and decoded images side by side
def display_images(original_image_path, decoded_tensor):
    original_image = Image.open(original_image_path)
    decoded_image = transforms.ToPILImage()(decoded_tensor.squeeze(0))  # Remove batch dimension

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide axes

    axes[1].imshow(decoded_image)
    axes[1].set_title('Decoded Image')
    axes[1].axis('off')  # Hide axes

    plt.show()

# Directory containing images
train_data_dir = '/Users/shrijansshetty/dev/AI/Source/AIImageClassifier/train_data'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(train_data_dir) if f.endswith('.jpg')]

# Randomly select 5 images
selected_images = random.sample(image_files, 5)

# Process each selected image
for image_file in selected_images:
    image_path = os.path.join(train_data_dir, image_file)
    
    # Preprocess the image
    input_img = preprocess_image(image_path)
    
    # Encode and decode the image
    with torch.no_grad():  # No need to track gradients
        encoded, decoded = autoencoder(input_img)
        print(f"Encoded output size for {image_file}: {encoded.size()}")  # Print the size of the encoded output
    
    # Display the original and decoded images
    display_images(image_path, decoded)