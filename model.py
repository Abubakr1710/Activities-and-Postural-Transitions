from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, seq_len):
        super().__init__()
        self.num_layer =num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.batch_size*self.hidden_size, seq_len)
        self.fc2 = nn.Linear(seq_len, self.batch_size*self.seq_len)

    def forward(self,x):
        h_0 = torch.zeros((self.num_layer, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((self.num_layer, self.batch_size, self.hidden_size))
        
        lstm_out, (h_n, c_n) = self.lstm(x,(h_0,c_0))
        last_hidden = h_n[-1]

        x = F.relu(last_hidden.flatten())
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out
    
model = LSTM()