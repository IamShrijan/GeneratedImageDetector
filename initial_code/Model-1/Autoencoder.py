import torch 
import torch.nn as nn

class CNNAutoencoderWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNAutoencoderWithDropout, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 384, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 192, 128)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 192, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 192, 128),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 96, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 192, 128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 48, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 192, 128),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 96, 64)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 192, 128)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 384, 256)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 768, 512)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Example usage
dropout_rate = 0.5  # Specify the dropout rate
autoencoder = CNNAutoencoderWithDropout(dropout_rate)
input_img = torch.randn(1, 3, 768, 512)  # Example input
encoded, decoded = autoencoder(input_img)

# To get the encoded image for classification
encoded_image = encoded
        