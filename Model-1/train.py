import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from data_utils import get_data_loaders
from Autoencoder import CNNAutoencoderWithDropout
import torch.nn as nn
import torch.optim as optim
import torch

# Assuming the data_utils.py file is in the same directory or properly imported
# from AI.Source.AIImageClassifier.data_utils import get_data_loaders
def train_autoencoder(csv_file, num_epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.5):
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((768, 512)),
        transforms.ToTensor()
    ])

    # Create data loaders
    train_loader, test_loader = get_data_loaders(csv_file, batch_size=batch_size, transform=transform)

    # Initialize the autoencoder
    autoencoder = CNNAutoencoderWithDropout(dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        for images, _ in train_loader:  # Labels are not needed for autoencoder training
            optimizer.zero_grad()
            _, decoded = autoencoder(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Running loss: {running_loss:.4f}')

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        # Save the model weights after each epoch
        torch.save(autoencoder.state_dict(), f'autoencoder_epoch_{epoch+1}.pth')

    print('Training complete.')

train_autoencoder('/Users/shrijansshetty/dev/AI/Source/AIImageClassifier/train.csv', num_epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.5)
        