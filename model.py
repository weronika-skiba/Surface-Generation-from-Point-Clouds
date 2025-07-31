import torch.nn as nn
import torch
import torch.nn.functional as F

### 
import torch.nn.utils as utils

class ReusableBlock(nn.Module):
    def __init__(self, input_size, hidden_size, use_norm, use_dropout, dropout_probability):
        super(ReusableBlock, self).__init__()

        self.use_norm = use_norm
        self.use_dropout = use_dropout

        self.fc = utils.weight_norm(nn.Linear(input_size, hidden_size)) if self.use_norm else nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_probability)
        
    def forward(self, x):
        # Forward pass through the block
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x) if self.use_dropout else x
        return x
    
###

class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        ##########################################################
        self.dims = dims
        self.dropout = set(dropout)
        self.dropout_prob = dropout_prob
        self.norm_layers = set(norm_layers)
        self.latent_in = set(latent_in)
        self.weight_norm = weight_norm
        self.use_tanh = use_tanh

        self.input_dim = 3
        self.blocks = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            in_dim = self.input_dim if i == 0 else self.dims[i]
            out_dim = self.dims[i+1] - self.input_dim if i + 1 in latent_in else self.dims[i+1]

            block = ReusableBlock(
                input_size = in_dim,
                hidden_size = out_dim,
                use_norm = i in self.norm_layers,
                use_dropout = i in self.dropout,
                dropout_probability = self.dropout_prob
            )

            self.blocks.append(block)
            
        self.fc = nn.Linear(self.dims[-1], 1)
        self.th = nn.Tanh()
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        out = input
        for i, block in enumerate(self.blocks):
            if i in self.latent_in:
                out = torch.cat([out, input], dim=1)  # inject original input
            out = block(out)
        out = self.fc(out)
        if self.use_tanh:
            out = self.th(out)
        ##########################################################

        return out
