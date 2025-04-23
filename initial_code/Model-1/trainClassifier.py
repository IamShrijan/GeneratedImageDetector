import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from data_utils import get_data_loaders
from Autoencoder import CNNAutoencoderWithDropout
import torch.nn as nn
import torch.optim as optim
import torch
from BinaryClassifier import BinaryClassifier

def train_autoencoder(csv_file, num_epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.5):
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((768, 512)),
        transforms.ToTensor()
    ])

    # Create data loaders
    train_loader, test_loader = get_data_loaders(csv_file, batch_size=batch_size, transform=transform, num_images=1024)

    # Initialize the autoencoder
    # Create an instance of the model
    autoencoder = CNNAutoencoderWithDropout()
    classifier = BinaryClassifier(393216)

    # Load the model weights
    state_dict = torch.load('autoencoder_epoch_1.pth',weights_only=True)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()  # Set the model to evaluation mode
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, y_actual in train_loader:  # Labels are not needed for autoencoder training
            print(images.shape)
            y_actual = y_actual.float()
            encoder, decoded = autoencoder(images)
            y_pred = classifier(encoder)
            loss = criterion(y_pred.flatten(), y_actual)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Calculate accuracy
            predicted = (y_pred > 0.5).float()  # Apply threshold for binary classification
            correct += (predicted.squeeze() == y_actual).sum().item()
            #print(predicted, y_actual)
            total += y_actual.size(0)
            #print(predicted.shape, y_actual.shape, correct, total)

        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save the model weights after each epoch
        torch.save(classifier.state_dict(), f'classifier_epoch_{epoch+1}.pth')

    print('Training complete.')

    # Evaluate on the test data
    test_accuracy = evaluate_classifier(classifier, autoencoder, test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

def evaluate_classifier(classifier, autoencoder, test_loader):
    classifier.eval()  # Set classifier to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during evaluation
        for images, y_actual in test_loader:
            encoder, _ = autoencoder(images)  # Only use the encoder
            y_pred = classifier(encoder)
            predicted = (y_pred > 0.5).float()  # Apply threshold for binary classification
            correct += (predicted.squeeze() == y_actual).sum().item()
            total += y_actual.size(0)

    accuracy = 100 * correct / total
    return accuracy

#train_autoencoder('/Users/shrijansshetty/dev/AI/Source/AIImageClassifier/train.csv', num_epochs=10, batch_size=32, learning_rate=0.001, dropout_rate=0.5)
transform = transforms.Compose([
        transforms.Resize((768, 512)),
        transforms.ToTensor()
    ])
train_loader, test_loader = get_data_loaders('/Users/shrijansshetty/dev/AI/Source/AIImageClassifier/train.csv', batch_size=32, transform=transform, num_images=1024)
state = torch.load('classifier_epoch_10.pth', weights_only=True)
classifier = BinaryClassifier(393216)
classifier.load_state_dict(state_dict=state)
state_dict = torch.load('autoencoder_epoch_1.pth',weights_only=True)
autoencoder = CNNAutoencoderWithDropout()
autoencoder.load_state_dict(state_dict)
autoencoder.eval()  # Set the model to evaluation mode
test_accuracy = evaluate_classifier(classifier=classifier,autoencoder=autoencoder, test_loader=test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')