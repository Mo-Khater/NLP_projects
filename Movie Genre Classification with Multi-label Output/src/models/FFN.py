import torch 
import torch.nn as nn

class FNN_model(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))