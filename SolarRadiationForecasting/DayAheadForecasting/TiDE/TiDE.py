import torch
import torch.nn as nn

from TiDE.t2v import SineActivation
from TiDE.ResidualBlock import ResidualBlock
from TiDE.Components import Encoder, Decoder, TemporalDecoder

class TiDE(nn.Module):
    # need to change this to args
    def __init__(self, args):
        super(TiDE, self).__init__()

        self.proj_input_dim = args.proj_input_dim
        self.proj_hidden_dim = args.proj_hidden_dim
        self.proj_output_dim = args.proj_output_dim
        self.t2v_input_dim = args.t2v_input_dim
        self.t2v_output_dim = args.t2v_output_dim
        self.num_enc_layers = args.num_enc_layers
        self.num_dec_layers = args.num_dec_layers
        self.dec_output_dim = args.dec_output_dim
        self.enc_input_dims = args.enc_input_dims
        self.enc_hid_dims = args.enc_hidden_dims
        self.enc_out_dims = args.enc_out_dims
        self.dec_input_dims = args.dec_input_dims
        self.dec_hid_dims = args.dec_hidden_dims
        self.temporal_hidden_dim = args.temporal_hidden_dim
        self.L = args.L
        self.H = args.H
        self.batch_size = args.batch_size
        self.dec_out_dims = args.dec_out_dims
        self.dropout = args.dropout





        self.res_block = ResidualBlock(self.proj_input_dim, self.proj_hidden_dim, 
                                       self.proj_output_dim, self.dropout)
        
        self.t2v = SineActivation(self.t2v_input_dim, self.t2v_output_dim)

        self.encoder = Encoder(self.enc_input_dims, self.enc_hid_dims, 
                               self.enc_out_dims, self.num_enc_layers, self.dropout)
        
        self.decoder = Decoder(self.dec_input_dims, self.dec_hid_dims, 
                               self.dec_out_dims, self.num_dec_layers, self.dropout)
        
        self.temporal_decoder = TemporalDecoder(self.dec_output_dim + self.proj_output_dim + self.t2v_output_dim,
                                                self.temporal_hidden_dim, 1, self.dropout)

        # self.temporal_decoder = TemporalDecoder(self.dec_output_dim + self.t2v_output_dim,
                                                # self.temporal_hidden_dim, 1, self.dropout)

        self.global_residual = nn.Linear(self.L, self.H)



    def forward(self, y, xw, xt):
        xw_proj = self.res_block(xw)
        
        # xt_emb = self.t2v(xt.view(-1, xt.shape[-1]))

        # Reshape xw_proj and xt_emb back to their original shapes
        # xw_proj = xw_proj.view(*xw.shape[:-1], -1)
        # xt_emb = xt_emb.view(*xt.shape[:-1], -1)

        # Stack xw_proj and xt_emb along the feature axis to create X
        # X = torch.cat((xw_proj, xt_emb), dim=-1)
        
        X = torch.cat((xw_proj, xt), dim=-1)
        # X = xt

        # Flatten X along the sequence and feature axes
        X_flattened = X.view(X.shape[0], -1)

        # Flatten y along the sequence axis
        y_flattened = y.view(y.shape[0], -1)

        # Concatenate y_flattened and X_flattened to create encoding_input
        encoding_input = torch.cat((y_flattened, X_flattened), dim=-1)

        encoding_input = encoding_input.view(-1, 1, encoding_input.shape[1])

        e = self.encoder(encoding_input)

        g = self.decoder(e)

        # use encoding_input.size(0) instead of self.batch_size
        # in case last batch is less than normal batch_size
        D = g.view(encoding_input.size(0), self.dec_output_dim, self.H)

        y_pred = []
        for t in range(self.H):
            d_t = D[:, :, t]

            # we only want to append future covariates

            xw_proj_t = xw_proj[:, self.L + t, :]

            # xw_proj_t = xw[:, self.L + t, :]
            xt_emb_t = xt[:, self.L + t, :]

            temporal_input = torch.cat((d_t, xw_proj_t, xt_emb_t), dim=1)
            # temporal_input = torch.cat((d_t, xt_emb_t), dim=1)

            y_t = self.temporal_decoder(temporal_input.view(encoding_input.size(0), 1, temporal_input.size(1)))
            y_pred.append(y_t)

        y_pred = torch.cat(y_pred, dim=1)
        y_l =  y[:, :self.L]

        y_pred = y_pred.reshape(y_pred.size(0), y_pred.size(1))
        y_l = y_l.reshape(y_l.size(0), y_l.size(1))

        res = self.global_residual(y_l)
        y_pred += res

        return y_pred.unsqueeze(-1)
    
    # Don't forget to un-normalise y_pred AFTER training
    # So separately from training
    # via:
    # y_pred = y_pred * (original_max - original_min) + original_min

