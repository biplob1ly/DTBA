import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from constants import PADDING_INDEX


class CNNModel(nn.Module):
    def __init__(self, drug_charset_size, drug_embed_size, drug_kernel_size,
                 protein_charset_size, protein_embed_size, protein_kernel_size,
                 num_filters):
        super(CNNModel, self).__init__()
        self.drug_embedder = nn.Embedding(drug_charset_size+1, drug_embed_size, padding_idx=PADDING_INDEX)     # +1 for padding
        self.drug_cnn1 = nn.Conv1d(in_channels=drug_embed_size, out_channels=num_filters,
                                  kernel_size=drug_kernel_size, stride=1, padding=0)
        self.drug_cnn2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2,
                                  kernel_size=drug_kernel_size, stride=1, padding=0)
        self.drug_cnn3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3,
                                  kernel_size=drug_kernel_size, stride=1, padding=0)

        self.protein_embedder = nn.Embedding(protein_charset_size+1, protein_embed_size, padding_idx=PADDING_INDEX)     # +1 for padding
        self.protein_cnn1 = nn.Conv1d(in_channels=protein_embed_size, out_channels=num_filters,
                                    kernel_size=protein_kernel_size, stride=1, padding=0)
        self.protein_cnn2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2,
                                    kernel_size=protein_kernel_size, stride=1, padding=0)
        self.protein_cnn3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3,
                                    kernel_size=protein_kernel_size, stride=1, padding=0)

        self.fc1 = nn.Linear(num_filters*6, 1024)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, XD, XT):
        XD = self.drug_embedder(XD)     # B x DS -> B x DS x E
        XD = self.drug_cnn1(XD.permute(0, 2, 1))     # B x E x DS -> B x DF x (DS - K + 1)
        XD = self.drug_cnn2(XD)     # B x DF x (DS - K + 1) -> B x 2*DF x (DS - 2*K + 2)
        XD = self.drug_cnn3(XD)     # B x 2*DF x (DS - 2*K + 2) -> B x 3*DF x (DS - 3*K + 3)
        XD = F.adaptive_max_pool1d(F.relu(XD), 1)       # B x 3*DF x (DS - 3*K + 3) - > B x 3*DF x 1

        XT = self.protein_embedder(XT)     # B x DS -> B x TS x E
        XT = self.protein_cnn1(XT.permute(0, 2, 1))     # B x E x TS -> B x TF x (TS - K + 1)
        XT = self.protein_cnn2(XT)       # B x TF x (TS - K + 1) -> B x 2*TF x (TS - 2*K + 2)
        XT = self.protein_cnn3(XT)       # B x 2*TF x (TS - 2*K + 2) -> B x 3*TF x (TS - 3*K + 3)
        XT = F.adaptive_max_pool1d(F.relu(XT), 1)       # B x 3*TF x (TS - 3*K + 1) - > B x 3*TF x 1

        DTI = torch.cat((XD.squeeze(), XT.squeeze()), 1)    # (B x 3*DF x 1) + (B x 3*TF x 1) -> B x 6*F

        DTI = F.relu(self.fc1(DTI))     # B x 6*F -> B x 1024
        DTI = self.dropout1(DTI)
        DTI = F.relu(self.fc2(DTI))     # B x 1024 -> B x 1024
        DTI = self.dropout2(DTI)
        DTI = F.relu(self.fc3(DTI))     # B x 1024 -> B x 512
        out = self.output(DTI)          # B x 512 -> B x 1
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, drug_charset_size, drug_embed_size, max_drug_length,
                 protein_charset_size, protein_embed_size, max_protein_length,
                 num_layers, num_heads, forward_expansion, dropout_rate,
                 batch_size):
        super(Transformer, self).__init__()
        self.drug_embed_size = drug_embed_size
        self.protein_embed_size = protein_embed_size
        self.batch_size = batch_size
        self.drug_embedder = nn.Embedding(drug_charset_size+1, drug_embed_size, padding_idx=PADDING_INDEX)
        self.protein_embedder = nn.Embedding(protein_charset_size+1, protein_embed_size, padding_idx=PADDING_INDEX)
        self.drug_pos_encoder = PositionalEncoding(drug_embed_size, dropout_rate, max_drug_length)
        self.protein_pos_encoder = PositionalEncoding(protein_embed_size, dropout_rate, max_protein_length)
        drug_encoder_layer = TransformerEncoderLayer(d_model=drug_embed_size, nhead=num_heads,
                                                      dim_feedforward=forward_expansion * drug_embed_size,
                                                      dropout=dropout_rate)
        protein_encoder_layer = TransformerEncoderLayer(d_model=protein_embed_size, nhead=num_heads,
                                                        dim_feedforward=forward_expansion * protein_embed_size,
                                                        dropout=dropout_rate)
        self.drug_encoder = TransformerEncoder(drug_encoder_layer, num_layers)
        self.protein_encoder = TransformerEncoder(protein_encoder_layer, num_layers)

        self.fc1 = nn.Linear(drug_embed_size + protein_embed_size, 1024)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, drugs, proteins):
        d_padding_masks = (drugs == PADDING_INDEX).to(drugs.device)   # B x DS
        p_padding_masks = (proteins == PADDING_INDEX).to(proteins.device)   # B x TS

        d_embeds = self.drug_pos_encoder(self.drug_embedder(drugs) * math.sqrt(self.drug_embed_size))   # B x DS -> B x DS x E
        d_embeds = d_embeds.permute(1, 0, 2)    # DS x B x E
        p_embeds = self.protein_pos_encoder(self.protein_embedder(proteins) * math.sqrt(self.protein_embed_size))   # B x TS -> B x TS x E
        p_embeds = p_embeds.permute(1, 0, 2)    # TS x B x E

        encoded_drugs = self.drug_encoder(d_embeds, src_key_padding_mask=d_padding_masks)    # DS x B x E
        encoded_drugs = encoded_drugs.permute(1, 0, 2)   # B x DS x E
        pooled_drugs = encoded_drugs.mean(dim=1)
        encoded_proteins = self.protein_encoder(p_embeds, src_key_padding_mask=p_padding_masks)   # TS x B x E
        encoded_proteins = encoded_proteins.permute(1, 0, 2)   # B x TS x E
        pooled_proteins = encoded_proteins.mean(dim=1)

        DTI = torch.cat((pooled_drugs, pooled_proteins), 1)    # (B x E) + (B x E) -> B x 2*E

        DTI = F.relu(self.fc1(DTI))     # B x 2*E -> B x 1024
        DTI = self.dropout1(DTI)
        DTI = F.relu(self.fc2(DTI))     # B x 1024 -> B x 1024
        DTI = self.dropout2(DTI)
        DTI = F.relu(self.fc3(DTI))     # B x 1024 -> B x 512
        out = self.output(DTI)          # B x 512 -> B x 1
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, attn_dims, attn_expansion=2, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size*attn_expansion)
        self.tnh = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size*attn_expansion, attn_dims)

    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x H -> B x S x attn_expansion*H
        output_1 = self.tnh(self.l1(hidden))
        # output_1 = self.dropout(output_1)

        # output_2: B x S x attn_expansion*H -> B x S x attn_dims(O)
        output_2 = self.l2(output_1)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x attn_dims(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        weighted_output = weighted_output.sum(dim=1)   # B x O x H -> B x H
        return weighted_output, attn_weights
        

class TransDTBA(nn.Module):
    def __init__(self, drug_charset_size, drug_embed_size, max_drug_length,
                 protein_charset_size, protein_embed_size, max_protein_length,
                 num_layers, num_heads, forward_expansion, dropout_rate,
                 batch_size, attn_dims=64):
        super(TransDTBA, self).__init__()
        self.drug_embed_size = drug_embed_size
        self.protein_embed_size = protein_embed_size
        self.batch_size = batch_size
        self.drug_embedder = nn.Embedding(drug_charset_size+1, drug_embed_size, padding_idx=PADDING_INDEX)
        self.protein_embedder = nn.Embedding(protein_charset_size+1, protein_embed_size, padding_idx=PADDING_INDEX)
        self.drug_pos_encoder = PositionalEncoding(drug_embed_size, dropout_rate, max_drug_length)
        self.protein_pos_encoder = PositionalEncoding(protein_embed_size, dropout_rate, max_protein_length)
        drug_encoder_layer = TransformerEncoderLayer(d_model=drug_embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*drug_embed_size, dropout=dropout_rate)
        protein_encoder_layer = TransformerEncoderLayer(d_model=protein_embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*protein_embed_size, dropout=dropout_rate)
        self.drug_encoder = TransformerEncoder(drug_encoder_layer, num_layers)
        self.protein_encoder = TransformerEncoder(protein_encoder_layer, num_layers)

        self.drug_attn_layer = Attention(drug_embed_size, attn_dims)
        self.protein_attn_layer = Attention(protein_embed_size, attn_dims)

        self.output = nn.Linear(drug_embed_size + protein_embed_size, 1)

    def forward(self, drugs, proteins):
        # attn_mask: B x S -> B x S x 1
        d_attn_mask = (drugs != PADDING_INDEX).unsqueeze(2).to(drugs.device)
        p_attn_mask = (proteins != PADDING_INDEX).unsqueeze(2).to(proteins.device)
        
        d_padding_masks = (drugs == PADDING_INDEX).to(drugs.device)   # B x DS
        p_padding_masks = (proteins == PADDING_INDEX).to(proteins.device)   # B x TS

        d_embeds = self.drug_pos_encoder(self.drug_embedder(drugs) * math.sqrt(self.drug_embed_size))   # B x DS x E
        d_embeds = d_embeds.permute(1, 0, 2)    # DS x B x E
        p_embeds = self.protein_pos_encoder(self.protein_embedder(proteins) * math.sqrt(self.protein_embed_size))   # B x TS x E
        p_embeds = p_embeds.permute(1, 0, 2)    # TS x B x E

        encoded_drugs = self.drug_encoder(d_embeds, src_key_padding_mask=d_padding_masks)    # DS x B x E
        encoded_drugs = encoded_drugs.permute(1, 0, 2)   # B x DS x E
        encoded_proteins = self.protein_encoder(p_embeds, src_key_padding_mask=p_padding_masks)   # TS x B x E
        encoded_proteins = encoded_proteins.permute(1, 0, 2)   # B x TS x E

        # both encoded_drugs and encoded_proteins are of shape: batch_size, seq_len, embed_size
        attended_drugs, drug_attn_weights = self.drug_attn_layer(encoded_drugs, d_attn_mask)
        attended_proteins, protein_attn_weights = self.protein_attn_layer(encoded_proteins, p_attn_mask)

        DTI = torch.cat((attended_drugs, attended_proteins), 1)
        out = self.output(DTI)
        return out
