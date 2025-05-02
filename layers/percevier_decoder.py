import torch
import torch.nn as nn

from layers.attention  import MultiHead_Attention, Encoder_Decoder_MultiHead_Attention

class Percevier_Decoder(nn.Module):
    def __init__(self, config):
        super(Percevier_Decoder, self).__init__()

        self.perciever_decoder_num_layers = config.perciever_decoder_num_layers

        self.from_embeddings = Encoder_Decoder_MultiHead_Attention(config.embed_dim, config.num_heads)
        self.layers = nn.ModuleList(
            [MultiHead_Attention(config.embed_dim, config.num_heads) for _ in range(self.perciever_decoder_num_layers)]
        )

    def forward(self, half_life, dna_position_embeddings):
        x, attention_scores = self.from_embeddings(half_life, dna_position_embeddings, dna_position_embeddings)

        for layer in self.layers:
            x, attention_scores = layer(x)
        return x