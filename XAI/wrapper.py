import torch
import torch.nn

class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input):

        emb, one_hot = input        # input: one-hot or embedded input
        return self.model(emb, one_hot)