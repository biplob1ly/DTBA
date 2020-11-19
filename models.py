import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, drug_charset_size, drug_embedding_dim, drug_kernel_size,
                 target_charset_size, target_embedding_dim, target_kernel_size,
                 num_filters):
        super(CNNModel, self).__init__()
        self.drug_embed = nn.Embedding(drug_charset_size, drug_embedding_dim)
        self.drug_cnn = nn.Conv1d(in_channels=drug_embedding_dim, out_channels=num_filters,
                                  kernel_size=drug_kernel_size, stride=1, padding=0)

        self.target_embed = nn.Embedding(target_charset_size, target_embedding_dim)
        self.target_cnn = nn.Conv1d(in_channels=target_embedding_dim, out_channels=num_filters,
                                    kernel_size=target_kernel_size, stride=1, padding=0)

        self.fc1 = nn.Linear(num_filters*2, 1024)
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


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size,
                 bidirect, dropout=.1, num_layers=1):
        """
        Even if we don't use it, it means we could potentially try 2 different models for 
        protein encoding. Also, it has fewer parameters than the transformer - meaning it 
        might work better for the smaller protein dataset.
        
        Model is LSTM followed by three linear/dropout layers
        """
        super(LSTM, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirect)
        self.linear_1 = nn.Linear(2*hidden_size, 512)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(512, 256)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_3 = nn.Linear(256, out_size)
    
    def forward(self, data):
        """
        data is [num_examples x max_example_length]
        returns vector of shape [num_examples x max_example_length x embedding_size]
        """
        # Hacky way to get the first zero in each example
        mask = [torch.where(i == 0)[0] for i in data]
        lengths = [x[0].item() if len(x) > 0 else data.shape[-1] for x in mask]
        
        embedding = self.embedder(data)
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedding, lengths, enforce_sorted=False, batch_first=True)
        out, hn = self.lstm(packed_embedding)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(out)
        
        output = F.relu(self.linear_1(out.data))
        output = F.relu(self.linear_2(self.dropout_1(output)))
        output = F.relu(self.linear_3(self.dropout_2(output)))
        
        return output.permute(1,0,2)


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
        input_size: embe
        n_head: number of heads in multihead attention
        hidden_size: size of embeddings/hidden layers
        n_layers: number of encoding layers in transformer
        """
        super(Transformer, self).__init__()
        self.encoder = nn.Embedding(num_embeddings, input_size)
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = TransformerEncoderLayer(input_size, n_head, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
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


class InteractionNetwork(nn.Module):
    def __init__(self, drug_model, target_model, drug_size, target_size):
        super(InteractionNetwork, self).__init__()
        self.target_model = target_model
        self.drug_model = drug_model
        self.embedding_size = drug_size + target_size
        
        self.dense_1 = nn.Linear(self.embedding_size, 1024)
        self.dropout_1 = nn.Dropout(0.1)
        self.dense_2 = nn.Linear(1024, 1024)
        self.dropout_2 = nn.Dropout(0.1)
        self.dense_3 = nn.Linear(1024, 512)
        
        self.prediction = nn.Linear(512, 1)
        
    def forward(self, drug, target):
        drug = self.drug_model(drug)
        target = self.target_model(target)
        embedding = torch.cat((drug, target), dim=1)
        
        output = F.relu(self.dense_1(embedding))
        output = self.dropout_1(output)
        output = F.relu(self.dense_2(output))
        output = self.dropout_2(output)
        output = F.relu(self.dense_3(output))
        output = self.prediction(output)    # Kernel_initializer = 'normal'
        