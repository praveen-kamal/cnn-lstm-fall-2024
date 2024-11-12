# evaluate.py

import torch
from torch.utils.data import DataLoader
from src.model import CNN_LSTM_Model  # Import your model class
from src.data_loader import get_dataloaders  # Import data loading functions
from src.utils import calculate_accuracy  # Import utility functions
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuration
batch_size = 512  # Same as used in training
num_classes = 10  # Update this to match the number of classes in your dataset
num_features = 9  # Update this to match the number of features in each sample
checkpoint_path = "output/best_model.pt"  # Path to the best saved model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load test data
_, test_loader = get_dataloaders(batch_size=batch_size, data_dir="data/processed/")

# Initialize model and load the best checkpoint
model = CNN_LSTM_Model(num_classes=num_classes, num_features=num_features).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()  # Set model to evaluation mode

# Evaluate on test set
all_labels = []
all_preds = []
total_test_loss, total_test_accuracy = 0, 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Accumulate loss and accuracy
        total_test_loss += loss.item()
        total_test_accuracy += calculate_accuracy(outputs, labels)

        # Store all labels and predictions for further evaluation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

# Calculate average test loss and accuracy
avg_test_loss = total_test_loss / len(test_loader)
avg_test_accuracy = (total_test_accuracy / len(test_loader)) * 100
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%")

# Generate a detailed classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
