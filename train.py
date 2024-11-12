# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_dataloaders
from src.model import CNN_LSTM_Model
from src.utils import save_model, calculate_accuracy, log_epoch, adjust_learning_rate
from src.config import (
    num_epochs,
    learning_rate,
    batch_size,
    num_classes,
    checkpoint_dir,
    decay_rate,
    decay_epoch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data loaders
train_loader, test_loader = get_dataloaders(
    batch_size=batch_size, data_dir="data/processed/"
)

# Initialize model, loss function, optimizer
num_features = 9
model = CNN_LSTM_Model(num_classes=num_classes, num_features=num_features).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}/{num_epochs}")

    # Training Phase
    total_loss, total_accuracy = 0, 0
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"Epoch: {epoch + 1} | Step: {i + 1}/{len(train_loader)}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = (total_accuracy / len(train_loader)) * 100
    print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%")
    log_epoch(epoch + 1, avg_train_loss, avg_train_accuracy)

    # Evaluation Phase
    model.eval()  # Set model to evaluation mode
    total_val_loss, total_val_accuracy = 0, 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss.item()
            total_val_accuracy += calculate_accuracy(outputs, labels)

    avg_val_loss = total_val_loss / len(test_loader)
    avg_val_accuracy = (total_val_accuracy / len(test_loader)) * 100
    print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.2f}%")

    # Adjust Learning Rate
    adjust_learning_rate(optimizer, epoch, learning_rate, decay_rate, decay_epoch)

    # Save model checkpoint periodically
    if (epoch + 1) % 5 == 0:
        save_model(
            model, optimizer, epoch + 1, f"{checkpoint_dir}/model_epoch_{epoch + 1}.pt"
        )
