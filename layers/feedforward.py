import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input):
        input = self.gelu(self.linear_1(input))
        input = self.linear_2(input)
        x = self.dropout(input)

        return x