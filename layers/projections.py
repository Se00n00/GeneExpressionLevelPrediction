import torch
import torch.nn as nn
import torch.nn.functional as F

class MRNAProjection(nn.Module):
    def __init__(self, input_features=8, seq_length=200, d_model=256):
        super(MRNAProjection, self).__init__()
        
        self.feature_projection = nn.Linear(input_features, d_model)
        
        # Parameters for interpolation
        self.seq_length = seq_length
        self.d_model = d_model
        
    def forward(self, mrna_features): #[batch, 8]
        
        # Step 1: Project features to d_model dimension
        # Output shape: [batch, d_model]
        projected = self.feature_projection(mrna_features)
        
        # Step 2: Expand to add a sequence dimension
        # Output shape: [batch, 1, d_model]
        projected = projected.unsqueeze(1)
        
        # Step 3: Repeat or interpolate to match sequence length
        # Option A: Simple repeat (replicate the same features across all positions)
        expanded = projected.repeat(1, self.seq_length, 1)
        
        # Alternative Option B: Use interpolation for smoother expansion
        # batch_size = mrna_features.shape[0]
        # projected_reshaped = projected.reshape(batch_size, 1, self.d_model)
        # expanded = F.interpolate(
        #     projected_reshaped.transpose(1, 2),  # [batch, d_model, 1]
        #     size=self.seq_length,
        #     mode='linear',
        #     align_corners=False
        # ).transpose(1, 2)  # [batch, seq_length, d_model]
        
        return expanded  # Shape: [batch, 20000, d_model]
    
class DNAProjection(nn.Module):
    def __init__(self, input_features=4, seq_length=200, d_model=256):
        super(MRNAProjection, self).__init__()
        
        self.feature_projection = nn.Linear(input_features, d_model)
        
        # Parameters for interpolation
        self.seq_length = seq_length
        self.d_model = d_model
        
    def forward(self, dna_features): #[batch, 2000, 4]
        DNA_One_hot = DNA_One_hot.permute(0, 2, 1)  # [B, 4, 20000]
        # Step 1: Project features to d_model dimension
        # Output shape: [batch, d_model]
        projected = self.feature_projection(mrna_features)
        
        # Step 2: Expand to add a sequence dimension
        # Output shape: [batch, 1, d_model]
        projected = projected.unsqueeze(1)
        
        # Step 3: Repeat or interpolate to match sequence length
        # Option A: Simple repeat (replicate the same features across all positions)
        expanded = projected.repeat(1, self.seq_length, 1)
        
        # Alternative Option B: Use interpolation for smoother expansion
        # batch_size = mrna_features.shape[0]
        # projected_reshaped = projected.reshape(batch_size, 1, self.d_model)
        # expanded = F.interpolate(
        #     projected_reshaped.transpose(1, 2),  # [batch, d_model, 1]
        #     size=self.seq_length,
        #     mode='linear',
        #     align_corners=False
        # ).transpose(1, 2)  # [batch, seq_length, d_model]
        
        return expanded  # Shape: [batch, 20000, d_model]