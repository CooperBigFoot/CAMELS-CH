import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train_model():
    # Model parameters
    input_size = 10  # Number of input features
    hidden_size = 64  # Number of features in hidden state
    num_layers = 2  # Number of stacked LSTM layers
    output_size = 1  # Number of output features

    # Create model instance
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop (dummy data for example)
    batch_size = 32
    seq_length = 15

    for epoch in range(10):
        # Generate dummy data
        x = torch.randn(batch_size, seq_length, input_size)
        y = torch.randn(batch_size, output_size)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_model()
