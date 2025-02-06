import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out
    
if __name__ == "__main__":

    input_size = 1       # e.g., one feature in time series
    hidden_size = 50
    num_layers = 1
    output_size = 1      # predicting a single value
    num_epochs = 100
    learning_rate = 0.001

    # Instantiate the model, loss function, and optimizer
    model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)

    # Print the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of trainable parameters: {total_params}")
