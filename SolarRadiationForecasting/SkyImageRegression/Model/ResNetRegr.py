import torch
import torch.nn as nn
import torchvision.models as models

from Model.t2v import SineActivation

class ResNetRegr(nn.Module):
    def __init__(self, output_dim, blocks, pretrained=True):
        super(ResNetRegr, self).__init__()

        self.t2v_input = 3
        self.t2v_output = 3
        self.t2v = SineActivation(self.t2v_input, self.t2v_output)

        # Load the desired ResNet model
        self.resnet = models.resnet18(pretrained=pretrained)
        # self.resnet = models.vit_b_16(weights='DEFAULT')

        # for param in self.resnet.parameters():
        #     param.requires_grad = True

        # Remove the original classification layer from ResNet

         # For ResNet-4: Only keep layer1 (2 blocks)
        if blocks == 4:
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-5])  # Removing layer2, layer3, layer4, avgpool, and fc
            linear_input_dim = 64
        # For ResNet-8: Only keep layer1 and layer2 (4 blocks)
        elif blocks == 8:
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-4])  # Removing layer3, layer4, avgpool, and fc
            linear_input_dim = 128
        # For ResNet-12: Keep layer1, layer2, and layer3 (6 blocks)
        elif blocks == 12:
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-3])  # Removing layer4, avgpool, and fc
            linear_input_dim = 256
        # Attach the modified layers to the custom model
        else:
            linear_input_dim = 512
        
        self.resnet = nn.Sequential(self.resnet, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(linear_input_dim, output_dim))






        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
       
        
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

        resnet_output = self.resnet(x_img)
        # print(resnet_output.shape, xt_emb.shape)
        concatenated = torch.cat([resnet_output, xt_emb], dim=-1)
        # concatenated = torch.cat([flattened_output, xt_emb], dim=-1)

        # Pass the concatenated tensor through the final forecast linear layer
        forecasts = self.fc_forecast(concatenated)

        return forecasts
        # return concatenated
    
