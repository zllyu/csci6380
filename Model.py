import torch
import torch.nn as nn


class Custom(nn.Module):
    def __init__(self, input_size):
        super(Custom, self).__init__()
        self.input_size = input_size
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=3),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.linear(x)

        return x.squeeze(0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_list=[], dropout=0.1):
        super(MLP, self).__init__()

        layers = nn.Sequential()
        for i, os in enumerate(hidden_list):
            layers.add_module(str(i * 2), nn.Linear(input_size, os))
            layers.add_module(str(i * 2 + 1), nn.ReLU())
            input_size = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0)

        x = self.hiddenLayers(x)
        x = self.out(self.dropout(x))

        return x.squeeze(0)
