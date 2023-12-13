import torch
import torch.nn as nn
import torchvision.models as models

from Model.t2v import SineActivation
from Model.ResNetRegr import ResNetRegr

class ResNet(nn.Module):
    def __init__(self, output_dim, hidden_dim, L, H, pretrained=True):
        super(ResNet, self).__init__()

        self.t2v_input = 3
        self.t2v_output = 3
        # self.t2v = SineActivation(self.t2v_input, self.t2v_output)

        # Load the desired ResNet model
        # self.resnet = models.resnet18(pretrained=pretrained)
        self.model = ResNetRegr(1024, False)
        # checkpoint = torch.load('./checkpoints/BestResNetRegr/checkpoint.pth', map_location=torch.device('cpu'))
        checkpoint = torch.load('./checkpoints/BestResNetRegr/checkpoint.pth')
        self.model.load_state_dict(checkpoint)

        # print("Original Model:")
        # print(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        # Remove the original classification layer from ResNet
        # self.model.resnet.fc = nn.Linear(self.model.resnet.fc.in_features, output_dim)

        # # Define the linear layer for the final forecast
        # self.fc_forecast = nn.Linear((output_dim*L) + L + (self.t2v_output*L), H)

        # Define the linear layer for the final forecast
        # self.fc_linear = nn.Linear((output_dim*L) + (self.t2v_output*L), hidden_dim)
        # self.fc_forecast = nn.Linear(hidden_dim, H)

        # Separate linear layers for each minute
        # self.fc_linear = nn.ModuleList([nn.Linear((output_dim*L) + L + (self.t2v_output*L), hidden_dim) for _ in range(H)])
        # self.model.fc_forecast = nn.Linear((output_dim*L) + L + (self.t2v_output*L), hidden_dim)
        self.fc_linear = nn.Linear((output_dim*L) + L + (self.t2v_output*L), hidden_dim)
        self.relu = nn.ReLU()
        self.fc_forecast = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(H)])


    def forward(self, yl, x_images, xt):
        yl = yl.view(yl.shape[0], -1)

        # xt_emb = self.t2v(xt.view(-1, xt.shape[-1]))
        # xt_emb = xt_emb.view(*xt.shape[:-1], -1)
        # xt_emb = xt_emb.view(xt_emb.shape[0], -1)

        # Pass the input images through the modified ResNet
        x_images = x_images.permute(1, 0, 2, 3, 4)
        xt = xt.permute(1,0,2)
        resnet_outputs = []
        # print(x_images.shape, xt.shape)
        for i in range(len(x_images)):
            resnet_output = self.model(x_images[i], xt[i])
            resnet_output = resnet_output.view(resnet_output.size(0), -1)  # Flatten the output
            resnet_outputs.append(resnet_output)

        # Stack the outputs along the temporal dimension to create time-distributed effect
        time_distributed_output = torch.stack(resnet_outputs, dim=1)
        flattened_output = time_distributed_output.view(time_distributed_output.size(0), -1)
        # Concatenate the time-distributed output with the lookback ghi values and time embeddings

        concatenated = torch.cat([flattened_output, yl], dim=-1)
        
        # concatenated = torch.cat([flattened_output, xt_emb], dim=-1)

        # Pass the concatenated tensor through the final forecast linear layer
        # output = self.fc_linear(concatenated)
        # forecasts = self.fc_forecast(output).unsqueeze(-1)
        # # forecasts = self.fc_forecast(concatenated).unsqueeze(-1)
        # return forecasts
    
        # Separate linear layers
        output = self.relu(self.fc_linear(concatenated))
        forecasts = []
        for i in range(len(self.fc_forecast)):
            # output = self.fc_linear[i](concatenated)
            forecast = self.fc_forecast[i](output).unsqueeze(-1)
            forecasts.append(forecast)

        return torch.cat(forecasts, dim=1)
