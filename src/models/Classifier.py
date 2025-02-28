import torch
import torch.nn as nn 

class ImageEmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=512, num_classes=14):
        super(ImageEmbeddingClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, hidden_dim)  # Last hidden layer (512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)  # Output layer

    def forward(self, x, return_hidden=False):
        x = self.relu1(self.fc1(x))
        last_layer = self.relu2(self.fc2(x))  # Extract last layer
        output = self.fc3(last_layer)  # Classification output

        if return_hidden:
            return last_layer  # Return the last hidden layer
        return output  # Default behavior