import torch
import torch.nn as nn
import torch.nn.functional as F


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




