import torch 
from torch import nn
class lstm_model(torch.nn.Module):
    def __init__(self,vocab_dim, embedded_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.embedded = nn.Embedding(vocab_dim, embedded_dim,padding_idx=0)
        self.lstm = torch.nn.LSTM(embedded_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedded(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
    