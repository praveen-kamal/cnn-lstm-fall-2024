# config.py

# Data parameters
input_size = 64  # Input dimension for images
seq_len = 10  # Number of frames in each sequence
num_classes = 10  # Number of classes in the AWID dataset

# Model parameters
hidden_size = 128  # Hidden layer size for LSTM
num_lstm_layers = 2  # Number of LSTM layers

# Training parameters
batch_size = 64
num_epochs = 20
learning_rate = 0.001
decay_rate = 0.1
decay_epoch = 10  # Decay the learning rate every 'decay_epoch' epochs

# Paths
data_dir = "data/processed/"
checkpoint_dir = "output/checkpoints/"
