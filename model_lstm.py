import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        # x.size(0),
        # print(h0.shape)
        # print(c0.shape)

        # forward
        out, _ = self.lstm(x, (h0, c0))

        # decode
        # [:,-1,:]
        out = self.fc(out)
        return out