# model.py
import torch
import torch.nn as nn


class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes, num_features):
        super(CNN_LSTM_Model, self).__init__()

        self.num_features = num_features  # Store num_features as a class attribute

        # Adjusted CNN layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(1, 3), padding=(0, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1)
        )

        # MaxPooling layer with smaller kernel size to avoid output size of zero
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.relu = nn.ReLU()

        # LSTM layer - adjust input size based on CNN output
        self.lstm = nn.LSTM(
            input_size=32 * (self.num_features // 4),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input: (batch_size, seq_len, channels=1, height=1, width=num_features)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, 1, 1, self.num_features)

        cnn_out = []
        for t in range(seq_len):
            out = self.conv1(x[:, t, :, :, :])  # Convolution
            out = self.relu(out)
            out = self.pool(out)  # Pooling
            out = self.conv2(out)
            out = self.relu(out)
            out = self.pool(out)
            cnn_out.append(out.view(batch_size, -1))  # Flatten for LSTM

        # Stack CNN outputs and pass through LSTM
        cnn_out = torch.stack(
            cnn_out, dim=1
        )  # Shape: (batch_size, seq_len, feature_dim)
        lstm_out, _ = self.lstm(
            cnn_out
        )  # LSTM expects (batch_size, seq_len, input_size)

        # Final fully connected layer for classification
        out = self.fc(lstm_out[:, -1, :])
        return out
