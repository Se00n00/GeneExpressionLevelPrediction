import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embed_dim, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.WQ = nn.Linear(in_features=embed_dim, out_features=model_dim)
        self.WK = nn.Linear(in_features=embed_dim, out_features=model_dim)
        self.WV = nn.Linear(in_features=embed_dim, out_features=model_dim)
        
    def forward(self, embeddings):
        Q = self.WQ(embeddings)
        K = self.WK(embeddings)
        V = self.WV(embeddings)
        
        # Scaled Dot-Product Attention
        dot_product = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1))
        scaled_dot_product = dot_product / torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float))
        attention_scores = F.softmax(scaled_dot_product, dim=-1)
        attention = torch.matmul(attention_scores, V)
        
        return attention, attention_scores

class MultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads to ensure proper concatenation"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model_dim = embed_dim // num_heads
        self.WO = nn.Linear(self.embed_dim, self.embed_dim)
        self.multi_heads = nn.ModuleList([
            Head(self.embed_dim, self.model_dim) for _ in range(self.num_heads)
        ])
        
    def forward(self, hidden_embeddings):
        head_outputs = []
        attention_scores_list = []
        
        for head in self.multi_heads:
            output, attention_scores = head(hidden_embeddings)
            head_outputs.append(output)
            attention_scores_list.append(attention_scores)
            
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.WO(multi_head_output)
        
        return output, attention_scores_list

class Encoder_Decoder_Head(nn.Module):
    def __init__(self, embed_dim, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.WQ = nn.Linear(in_features=embed_dim, out_features=model_dim)
        self.WK = nn.Linear(in_features=embed_dim, out_features=model_dim)
        self.WV = nn.Linear(in_features=embed_dim, out_features=model_dim)
        
    def forward(self, embeddings_q, embeddings_k, embeddings_v):
        Q = self.WQ(embeddings_q)
        K = self.WK(embeddings_k)
        V = self.WV(embeddings_v)
        
        # Scaled Dot-Product Attention
        dot_product = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1))
        scaled_dot_product = dot_product / torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float))
        attention_scores = F.softmax(scaled_dot_product, dim=-1)
        attention = torch.matmul(attention_scores, V)
        
        return attention, attention_scores

class Encoder_Decoder_MultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads to ensure proper concatenation"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model_dim = embed_dim // num_heads
        self.WO = nn.Linear(self.embed_dim, self.embed_dim)
        self.multi_heads = nn.ModuleList([
            Encoder_Decoder_Head(self.embed_dim, self.model_dim) for _ in range(self.num_heads)
        ])
        
    def forward(self, embeddings_q, embeddings_k, embeddings_v):
        head_outputs = []
        attention_scores_list = []
        
        for head in self.multi_heads:
            output, attention_scores = head(embeddings_q, embeddings_k, embeddings_v)
            head_outputs.append(output)
            attention_scores_list.append(attention_scores)
            
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.WO(multi_head_output)
        
        return output, attention_scores_list

# Example usage:
# embed_dim = 512  # Embedding dimension
# num_heads = 8    # Number of attention heads (must ensure embed_dim % num_heads == 0)