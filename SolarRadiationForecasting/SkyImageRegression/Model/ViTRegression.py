import torch
import torch.nn as nn
import torchvision.models as models

from Model.t2v import SineActivation

class ViTRegression(nn.Module):
    def __init__(self, output_dim, pretrained=True):
        super(ViTRegression, self).__init__()

        self.t2v_input = 3
        self.t2v_output = 3
        self.t2v = SineActivation(self.t2v_input, self.t2v_output)

        # Load the desired ResNet model

        # self.vit = models.vit_b_16(weights='DEFAULT')
        self.vit = models.vit_b_16(pretrained=pretrained)

        for param in self.vit.parameters():
            param.requires_grad = True

        # Remove the original classification layer from ResNet
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, output_dim)

        # # Define the linear layer for the final forecast
        self.fc_forecast = nn.Linear((output_dim) + (self.t2v_output), 1)

        # Define the linear layer for the final forecast
        # self.fc_linear = nn.Linear((output_dim*L) + (self.t2v_output*L), hidden_dim)
        # self.fc_forecast = nn.Linear(hidden_dim, H)

    def forward(self, x_img, xt):

        xt_emb = self.t2v(xt.view(-1, xt.shape[-1]))
        xt_emb = xt_emb.view(*xt.shape[:-1], -1)
        xt_emb = xt_emb.view(xt_emb.shape[0], -1)

        # Pass the input images through the modified ResNet

        vit_output = self.vit(x_img)

        concatenated = torch.cat([vit_output, xt_emb], dim=-1)
        # concatenated = torch.cat([flattened_output, xt_emb], dim=-1)

        # Pass the concatenated tensor through the final forecast linear layer
        forecasts = self.fc_forecast(concatenated)

        return forecasts
    
