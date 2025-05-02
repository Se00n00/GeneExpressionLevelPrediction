import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import Embeddings
from layers.projections import MRNAProjection
from layers.percevier_decoder import Percevier_Decoder
from layers.attention import Encoder_Decoder_Head
from layers.convo_attention import N_Convo_attention, conv1d_attention

# Config:
#   embed_dim
#   output_dim
# [Embeddings]
#   vocab_size
#   embed_dim
#   max_position_embeddings
#   dropout_prob = 1
# [Perceiver Decoder]
#   perceiver_decoder_num_layers
# [Convo Attention]
#   num_covo_attention_layers       # N Layers
#   conv_in_convo_attention
#   conv_out_convo_attention


# Missing:
#   Architectural Non-Essential: Activation (GeLU,), Normallization, Dropout

class Perceiver(nn.Module):
    def __init__(self, config, ConvoAttentionConfig, conv1d_attention_config):
        super(Perceiver, self).__init__()

        # Embedding Block
        self.Embedding_layer = Embeddings(
            config.vocab_size, 
            config.embed_dim,
            config.max_position_embeddings,
            config.dropout_prob
            ) # This Would add the positional information to Embeddings

        self.project_mrna = MRNAProjection()
        self.Decoder_layer = Percevier_Decoder(config)
        
        self.conv1d_attention = conv1d_attention(
            conv1d_attention_config.in_channels,
            conv1d_attention_config.out_channels,
            conv1d_attention_config.kernel_size,
            conv1d_attention_config.stride,
            conv1d_attention_config.pool_kernel_size,
            conv1d_attention_config.pool_stride,
            conv1d_attention_config.embed_dim,
            conv1d_attention_config.model_dim
            )
        self.conv_attention_layers = N_Convo_attention(ConvoAttentionConfig)
        self.fcl1 = nn.Linear(4, config.embed_dim)
        self.fcl2 = nn.Linear(64, config.embed_dim)

        self.conv_attention_decoder = Encoder_Decoder_Head(config.embed_dim, config.model_dim)

        self.final_linear = nn.Linear(2*config.embed_dim, 1)
    

    def forward(self, half_life, DNA_Ordinal, DNA_One_hot_encoded):

        # Embeddings Block
        DNA_position_embeddings = self.Embedding_layer(DNA_Ordinal)
        Expanded_Half_Life = self.project_mrna(half_life)
        DNA_contextual_embeddings = self.Decoder_layer(Expanded_Half_Life, DNA_position_embeddings)
        averaged_Contextual_embeddings = torch.mean(DNA_contextual_embeddings, dim=1)
        

        # Convolutional Attention Block
        conved = self.conv1d_attention(DNA_One_hot_encoded)
        convo_attention_pooled = self.conv_attention_layers(conved)
        reshaped = convo_attention_pooled.permute(0, 2, 1)
        x = self.fcl1(reshaped)

        # Attention Block for Convo-Attention and embeddings
        x, _ = self.conv_attention_decoder(x, DNA_contextual_embeddings, DNA_contextual_embeddings)
        x = self.fcl2(x)

        averaged_Convo_embeddings = torch.mean(x, dim=1)
        concatenated = torch.cat((averaged_Contextual_embeddings, averaged_Convo_embeddings),1)
        
        final_logits = self.final_linear(concatenated)

        return final_logits
