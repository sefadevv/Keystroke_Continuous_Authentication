import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            label.unsqueeze(1) * torch.pow(euclidean_distance, 2) +
            (1 - label.unsqueeze(1)) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class SiameseNetwork_V6(nn.Module):
    def __init__(self, vocab_size, dropout_rate=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        
        self.lstm = nn.LSTM(
            input_size=34, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.bn = nn.BatchNorm1d(128) 
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_embed = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
    def forward_once(self, x):
        emb = self.embedding(x[:, :, 2].long())
        out, _ = self.lstm(torch.cat((x[:, :, :2], emb), dim=2))
        
        last_hidden = out[:, -1, :] 
        normed = self.bn(last_hidden)
        embedded = self.fc_embed(self.dropout(normed))
        
        embedded = F.normalize(embedded, p=2, dim=1)
        return embedded

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2