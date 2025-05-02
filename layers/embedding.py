import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position_embeddings, dropout_prob, reduction_factor=100):
        super(Embeddings, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim).to("cuda")
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim).to("cuda")
        self.layer_norm = nn.LayerNorm(embed_dim, eps = 1e-12)
        self.dropout = nn.Dropout(dropout_prob)

        self.seq_reduction = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=reduction_factor,
            stride=reduction_factor,
            padding=0
        )

    def forward(self, input):
        seq_length = input.size(1)
        batch_size = input.size(0)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        token_embeddings = self.token_embeddings(input)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings     # Add Position Embeddings with token embeddings to include positional information

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Reduce sequence length using 1D convolution
        # [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        embeddings_transposed = embeddings.transpose(1, 2)
        
        # Apply convolution to reduce sequence length
        # [batch, embed_dim, seq_len] -> [batch, embed_dim, seq_len/reduction_factor]
        reduced_embeddings = self.seq_reduction(embeddings_transposed)
        
        # [batch, embed_dim, seq_len/reduction_factor] -> [batch, seq_len/reduction_factor, embed_dim]
        reduced_embeddings = reduced_embeddings.transpose(1, 2)
        
        return reduced_embeddings

