import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.input_size * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.input_size * 2, out_features=self.input_size * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.input_size * 4, out_features=3)
        )

    def forward(self, x):
        return self.linear(x)
