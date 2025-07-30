import torch.nn as nn
import torch
import torch.nn as nn

class FFNNModel(nn.Module): 
    def __init__(self, input_dim=300, hidden_dim=128, output_dim=1):
        super().__init__()  
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x



