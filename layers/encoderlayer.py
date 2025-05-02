import torch
import torch.nn as nn

from attention import MultiHead_Attention
from feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, pre_normallization=False):
        super(EncoderLayer, self).__init__()

        self.pre_normallization = pre_normallization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHead_Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForward(embed_dim=embed_dim, hidden_dim=ff_dim)
    
    def forward(self, x):

        if(self.pre_normallization): # Pre-Layer Normallization: Provides stable Training
            x = self.layer_norm1(x)
            x = x + self.attention(x)
            x = x + self.feed_forward(self.layer_norm2(x))
        else:                   # Post-Layer Normallization: It Requires Learning warm-up as gradients may diverge during training 
            x = x + self.attention(x)
            x = self.layer_norm1(x)
            x = x + self.feed_forward(x)
            x = self.layer_norm2(x)

        return x