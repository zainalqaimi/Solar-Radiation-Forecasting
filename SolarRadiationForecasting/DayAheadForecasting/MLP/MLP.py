import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        L = args.L
        H = args.H
        # input_size = args.L + ((args.L+args.H)*7)
        input_size = args.L + ((args.L+args.H)*2)

        # Define the layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, H)  # L+H timesteps, 8 features

    def forward(self, y, xw, xt):
        # Flatten inputY
        y = self.flatten(y)

        # Stack inputXw and inputXt
        # x = torch.cat([xw, xt], dim=-1)
        x = self.flatten(xt)

        # Concatenate the outputs
        combined = torch.cat([y, x], dim=-1)
        # print(combined.shape)

        # Apply the final linear layer
        outputs = self.fc(combined)

        return outputs.unsqueeze(-1)

# Instantiate the model
