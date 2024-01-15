import torch
import torch.nn as nn
import torch.optim as optim

# Define the binary classifier model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden and cell states to zero
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Extract the output from the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        # Apply sigmoid activation function
        out = self.sigmoid(out)

        return out
class Conv1DBinaryClassifier(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, output_size=1):
        super(Conv1DBinaryClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply 1D convolution
        x = self.conv1d(x)

        # Apply activation function
        x = self.relu(x)

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)

        # Apply sigmoid activation function
        x = self.sigmoid(x)

        return x