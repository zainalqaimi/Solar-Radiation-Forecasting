import torch 
import torch.nn as nn
import torch.nn.functional as F

from TiDE.ResidualBlock import ResidualBlock

# class Attention(nn.Module):
#     def __init__(self, memory_size, encoder_output_dim):
#         super(Attention, self).__init__()
#         # Need to figure out the memory matrix
#         # Currently randomly initialised and gets updated
#         # during backpropagation
#         self.memory = nn.Parameter(torch.randn((memory_size, encoder_output_dim)))

#         # we also want to add sparsity to the attention module

#     def forward(self, e):
#         attention_scores = F.softmax(e @ self.memory.T, dim=-1)
#         attention_output = attention_scores @ self.memory
#         em = torch.cat((e, attention_output), dim=1)
#         return em

class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_layers, dropout):
        super(Encoder, self).__init__()
        # self.layers = nn.ModuleList([ResidualBlock(input_dim, hidden_dim, output_dim) for _ in range(num_layers)])

        self.layers = nn.ModuleList([ResidualBlock(input_dims[i], 
                                    hidden_dims[i], output_dims[i], dropout) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_layers, dropout):
        super(Decoder, self).__init__()
        # self.layers = nn.ModuleList([ResidualBlock(input_dim, hidden_dim, output_dim) for _ in range(num_layers)])
        # self.fc = nn.Linear(input_dim, output_dim)

        self.layers = nn.ModuleList([ResidualBlock(input_dims[i], 
                                    hidden_dims[i], output_dims[i], dropout) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.fc(x)
        return x

class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TemporalDecoder, self).__init__()
        self.res_block = ResidualBlock(input_dim, hidden_dim, output_dim, dropout)
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.res_block(x)
        # x = self.fc(x)
        return x

