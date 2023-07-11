import torch

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_cells, num_layers, dropout=0):
        super(LSTM, self).__init__()
        self.drop = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_cells, num_layers)
        self.linear = nn.Linear(hidden_cells, 50)
        if dropout:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if self.drop:
            out, _ = self.dropout(self.lstm(x))
        else:    
            out, _ = self.lstm(x)
        out = self.linear(out)
        return out
    

class SiameseNet(nn.Module):
    def __init__(self, input_batch, encoder):
        super(SiameseNet, self).__init__()

        self.encoder = encoder
        self.linear = nn.Linear(input_batch, input_batch)
        self.sig = nn.Sigmoid()

    def manhattan_distance(self, q1, q2):
        return torch.exp(-torch.sum(torch.abs(q1 - q2))).cuda()    

    def forward(self, input1, input2):
        embed1 = self.encoder(input1)
        embed2 = self.encoder(input2)
        similarity_score = torch.zeros(embed1.size()[0]).cuda()
        # print(embed1.shape, embed2.shape, similarity_score.shape)
        for i in range(embed1.size()[0]):
            similarity_score[i] = self.manhattan_distance(embed1, embed2)        
        out = self.linear(similarity_score)
        return self.sig(out)
