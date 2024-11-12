import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# Model saving and loading
def save_model(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Model loaded from {path}, starting from epoch {start_epoch}")
    return model, optimizer, start_epoch


# Accuracy calculation
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


# Logging
def log_epoch(epoch, loss, accuracy):
    print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")


# Data normalization
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Learning rate adjustment
def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epoch=10):
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Learning rate adjusted to {lr}")


# Classification report and confusion matrix
def get_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
