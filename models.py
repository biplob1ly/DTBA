import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        self.drug_embed = nn.Embedding(args.drug_charset_size, args.drug_embedding_dim)
        self.drug_cnn = nn.Conv1d(in_channels=args.drug_embedding_dim, out_channels=args.num_filters,
                                  kernel_size=args.drug_kernel_size, stride=1, padding=0)

        self.target_embed = nn.Embedding(args.drug_charset_size, args.drug_embedding_dim)
        self.target_cnn = nn.Conv1d(in_channels=args.target_embedding_dim, out_channels=args.num_filters,
                                    kernel_size=args.target_kernel_size, stride=1, padding=0)

        self.fc1 = nn.Linear(args.num_filters*2, 1024)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, XD, XT):
        XD = self.drug_embed(XD)
        XD = self.drug_cnn(XD.permute(0, 2, 1))
        XD = F.adaptive_max_pool1d(F.relu(XD), 1)

        XT = self.target_embed(XT)
        XT = self.target_cnn(XT.permute(0, 2, 1))
        XT = F.adaptive_max_pool1d(F.relu(XT), 1)
        DTI = torch.cat((XD.squeeze(), XT.squeeze()), 1)

        DTI = F.relu(self.fc1(DTI))
        DTI = self.dropout1(DTI)
        DTI = F.relu(self.fc2(DTI))
        DTI = self.dropout2(DTI)
        out = self.output(DTI)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=.1, max_len=5000):
        """
        Taken from pytorch tutorial on Transformers
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        Which in turn seems to be taken from Zelun Wang
        https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        
        d_model: size of input vectors
        max_len: length of positions
        
        returns max_length * d_model position matrix
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        encoding = torch.zeros(max_len, d_model)
        position = torch.arage(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0).transpose(0,1)
        self.register_buffer('encoding', encoding)
    
    def forward(self, x):
        x = x + self.encoding[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, num_embeddings, input_size, n_head, hidden_size, n_layers, dropout=.1):
        """
        Taken from pytorch tutorial on Transformers
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        
        num_embeddings: Size of input dictionary
        input_size: Maximum size of inputs
        n_head: number of heads in multihead attention
        hidden_size: size of embeddings/hidden layers
        n_layers: number of encoding layers in transformer
        """
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = TransformerEncoderLayer(input_size, n_head, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(num_embeddings, input_size)
        self.input_size = input_size
        self.decoder = nn.Linear(input_size, num_embeddings)
        
        self.init_weights()
    
    def generate_square_subsequent_mask(self, size):
        """
        I'm actually not sure what purpose this function has
        But I copied it down anyway - maybe it solves a problem I don't understand yet.
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, x, x_mask):
        x = self.encoder(x) * math.sqrt(self.input_size)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, x_mask)
        output = self.decoder(x)
        return output


