import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


"""
Another Version of Transformers (from scratch) Begins from here
"""


class Embedder(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_length, dropout_rate):
        super(Embedder, self).__init__()
        self.token_embedder = nn.Embedding(vocab_size, embed_size)
        self.position_embedder = nn.Embedding(max_seq_length, embed_size)

        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tokens):
        seq_length = input_tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_tokens)

        token_embeddings = self.token_embedder(input_tokens)
        position_embeddings = self.position_embedder(position_ids)
        embeddings = self.dropout(token_embeddings + position_embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate, wide_attention=False):
        super(SelfAttention, self).__init__()

        self.num_heads = num_heads
        self.factor = 1
        if wide_attention:
            self.factor = num_heads

        if (self.factor * embed_size) % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")

        self.head_size = (self.factor * embed_size) // num_heads
        self.key_linear = nn.Linear(embed_size, self.factor * embed_size, bias=False)
        self.query_linear = nn.Linear(embed_size, self.factor * embed_size, bias=False)
        self.value_linear = nn.Linear(embed_size, self.factor * embed_size, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(self.factor * embed_size, embed_size)

    def attention(self, key, query, value, mask):
        # Following http://peterbloem.nl/blog/transformers

        key = key / (self.head_size ** (1/4))
        query = query / (self.head_size ** (1/4))
        scores = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, value)
        return output

    def forward(self, key, query, value, mask=None):
        batch_size, seq_len, embed_size = query.size()

        key = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_size)
        query = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_size)
        value = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_size)

        # Transpose to get dimensions batch_size x num_heads x seq_len x embed_size
        key = key.transpose(1, 2)  # .contiguous().view(batch_size * self.num_heads, seq_len, self.head_size)
        query = query.transpose(1, 2)  # .contiguous().view(batch_size * self.num_heads, seq_len, self.head_size)
        value = value.transpose(1, 2)  # .contiguous().view(batch_size * self.num_heads, seq_len, self.head_size)

        attended_output = self.attention(key, query, value, mask)
        # attended_output = attended_output.view(batch_size, self.num_heads, seq_len, self.head_size)

        unified_out = attended_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size) # embed_size = self.num_heads * self.head_size
        out = self.fc_out(unified_out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout_rate, wide_attention=False):
        super(TransformerBlock, self).__init__()
        self.self_attention = SelfAttention(embed_size, num_heads, dropout_rate, wide_attention)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, key, query, value, mask):
        attn_value = self.self_attention(key, query, value, mask)
        normalized = self.norm1(self.dropout(attn_value) + query)
        forwarded = self.feed_forward(normalized)
        out = self.norm2(self.dropout(forwarded) + normalized)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout_rate):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    forward_expansion,
                    dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden, mask):
        for layer in self.layers:
            hidden = layer(hidden, hidden, hidden, mask)

        return hidden


class TransDTBA(nn.Module):
    def __init__(self, drug_charset_size, drug_embed_size, max_drug_length,
                 protein_charset_size, protein_embed_size, max_protein_length,
                 num_layers, num_heads, forward_expansion, dropout_rate,
                 batch_size, num_filters=3, filter_size=3, dt_pad=0):
        super(TransDTBA, self).__init__()

        self.batch_size = batch_size
        self.dt_pad = dt_pad
        self.drug_embedder = Embedder(drug_charset_size, drug_embed_size, max_drug_length, dropout_rate)
        self.protein_embedder = Embedder(protein_charset_size, protein_embed_size, max_protein_length, dropout_rate)

        self.drug_encoder = Encoder(drug_embed_size, num_layers, num_heads, forward_expansion, dropout_rate)
        self.protein_encoder = Encoder(protein_embed_size, num_layers, num_heads, forward_expansion, dropout_rate)

        self.dti_cnn = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=filter_size)
        cout_h = max_drug_length - (num_filters - 1)
        cout_w = drug_embed_size - (num_filters - 1)
        cnn_out_size = num_filters * cout_h * cout_w
        self.ffnn = nn.Sequential(
            nn.Linear(cnn_out_size, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, drugs, proteins):
        d_masks = (drugs != self.dt_pad).unsqueeze(1).unsqueeze(2).to(drugs.device)
        t_masks = (proteins != self.dt_pad).unsqueeze(1).unsqueeze(2).to(proteins.device)

        d_embeds = self.drug_embedder(drugs)
        t_embeds = self.protein_embedder(proteins)

        encoded_drugs = self.drug_encoder(d_embeds, d_masks)
        encoded_proteins = self.protein_encoder(t_embeds, t_masks)

        # both encoded_drugs and encoded_proteins are of shape: batch_size, seq_len, embed_size
        b, s, e = encoded_drugs.size()
        d_mod = encoded_drugs.view(b, -1, s, e)
        t_mod = encoded_proteins.view(b, -1).unfold(1, s*e, e).view(b, -1, s, e)
        dti = torch.einsum('bdse, btse -> bdtse', [d_mod, t_mod])
        dti = dti.squeeze().sum(dim=-3)  # b x s x e

        dti = dti.unsqueeze(1)
        cnn_out = self.dti_cnn(dti)
        cnn_out_flattened = cnn_out.view(self.batch_size, -1)
        out = self.ffnn(cnn_out_flattened)

        return out



class TransDTBA(nn.Module):
    def __init__(self, drug_charset_size, drug_embed_size, max_drug_length,
                 protein_charset_size, protein_embed_size, max_protein_length,
                 num_layers, num_heads, forward_expansion, dropout_rate,
                 batch_size, num_filters=3, filter_size=3, dt_pad=0):
        super(TransDTBA, self).__init__()
        self.drug_embed_size = drug_embed_size
        self.protein_embed_size = protein_embed_size
        self.batch_size = batch_size
        self.dt_pad = dt_pad
        self.drug_embedder = nn.Embedding(drug_charset_size, drug_embed_size)
        self.protein_embedder = nn.Embedding(protein_charset_size, protein_embed_size)
        self.drug_pos_encoder = PositionalEncoding(drug_embed_size, dropout_rate, max_drug_length)
        self.protein_pos_encoder = PositionalEncoding(protein_embed_size, dropout_rate, max_protein_length)
        drug_encoder_layers = TransformerEncoderLayer(d_model=drug_embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*drug_embed_size, dropout=dropout_rate)
        protein_encoder_layers = TransformerEncoderLayer(d_model=protein_embed_size, nhead=num_heads,
                                                 dim_feedforward=forward_expansion*protein_embed_size, dropout=dropout_rate)
        self.drug_encoder = TransformerEncoder(drug_encoder_layers, num_layers)
        self.protein_encoder = TransformerEncoder(protein_encoder_layers, num_layers)

        self.dti_cnn = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=filter_size)
        cout_h = max_drug_length - (num_filters - 1)
        cout_w = drug_embed_size - (num_filters - 1)
        cnn_out_size = num_filters * cout_h * cout_w
        self.ffnn = nn.Sequential(
            nn.Linear(cnn_out_size, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, drugs, proteins):
        d_masks = (drugs == self.dt_pad).to(drugs.device)   # B x DS
        t_masks = (proteins == self.dt_pad).to(proteins.device)   # B x TS

        d_embeds = self.drug_pos_encoder(self.drug_embedder(drugs) * math.sqrt(self.drug_embed_size))   # B x DS x E
        d_embeds = d_embeds.permute(1, 0, 2)    # DS x B x E
        t_embeds = self.protein_pos_encoder(self.protein_embedder(proteins) * math.sqrt(self.protein_embed_size))   # B x TS x E
        t_embeds = t_embeds.permute(1, 0, 2)    # TS x B x E

        encoded_drugs = self.drug_encoder(d_embeds, src_key_padding_mask=d_masks)    # DS x B x E
        encoded_drugs = encoded_drugs.permute(1, 0, 2)   # B x DS x E
        encoded_proteins = self.protein_encoder(t_embeds, src_key_padding_mask=t_masks)   # TS x B x E
        encoded_proteins = encoded_proteins.permute(1, 0, 2)   # B x TS x E

        # both encoded_drugs and encoded_proteins are of shape: batch_size, seq_len, embed_size
        b, s, e = encoded_drugs.size()
        d_mod = encoded_drugs.reshape(b, -1, s, e)  # B x 1 x DS x E
        t_mod = encoded_proteins.reshape(b, -1).unfold(1, s*e, e).view(b, -1, s, e)  # B x (TS - DS + 1) x DS x E
        dti = torch.einsum('bdse, btse -> bdtse', [d_mod, t_mod])   # B x 1 x (TS - DS + 1) x DS x E
        dti = dti.squeeze().sum(dim=-3)  # B x DS x E

        dti = dti.unsqueeze(1)  # B x 1 x DS x E
        cnn_out = self.dti_cnn(dti)  # num_filters * cout_h * cout_w
        cnn_out_flattened = cnn_out.view(self.batch_size, -1)
        out = self.ffnn(cnn_out_flattened)
        return out


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, hidden_size,
                 bidirect, dropout=.1, num_layers=1):
        """
        Even if we don't use it, it means we could potentially try 2 different models for
        protein encoding. Also, it has fewer parameters than the transformer - meaning it
        might work better for the smaller protein dataset.

        Model is LSTM followed by three linear/dropout layers
        """
        super(LSTM, self).__init__()
        self.max_length = max_length
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirect)
        self.linear_1 = nn.Linear(2 * hidden_size, 512)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(512, 256)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_3 = nn.Linear(256, 1)

    def forward(self, data):
        """
        data is [num_examples x max_example_length]
        returns vector of shape [num_examples x max_example_length x embed_size]
        """
        # Hacky way to get the first zero in each example - could be more efficient
        mask = [torch.where(i == 0)[0] for i in data]
        lengths = [x[0].item() if len(x) > 0 else data.shape[-1] for x in mask]

        embedding = self.embedder(data)
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedding, lengths, enforce_sorted=False, batch_first=True)
        out, hn = self.lstm(packed_embedding)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(out, total_length=self.max_length)

        output = F.relu(self.linear_1(out.data))
        output = F.relu(self.linear_2(self.dropout_1(output)))
        output = F.relu(self.linear_3(self.dropout_2(output)))

        return output.permute(1, 0, 2)


class InteractionNetwork(nn.Module):
    def __init__(self, drug_model, protein_model, drug_output_size, protein_output_size):
        super(InteractionNetwork, self).__init__()
        self.protein_model = protein_model
        self.drug_model = drug_model
        self.data_size = drug_output_size + protein_output_size

        self.dense_1 = nn.Linear(self.data_size, 1024)
        self.dropout_1 = nn.Dropout(0.1)
        self.dense_2 = nn.Linear(1024, 1024)
        self.dropout_2 = nn.Dropout(0.1)
        self.dense_3 = nn.Linear(1024, 512)

        self.prediction = nn.Linear(512, 1)

    def forward(self, drug, protein):
        drug = self.drug_model(drug)
        protein = self.protein_model(protein)
        embedding = torch.cat((drug.squeeze(), protein.squeeze()), dim=1)

        output = F.relu(self.dense_1(embedding))
        output = self.dropout_1(output)
        output = F.relu(self.dense_2(output))
        output = self.dropout_2(output)
        output = F.relu(self.dense_3(output))
        output = self.prediction(output)  # Kernel_initializer = 'normal'
        return output
