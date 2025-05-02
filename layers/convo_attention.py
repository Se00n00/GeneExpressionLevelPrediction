import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import Head
#from layers.projections import DNA_Projections

# class Convo_attention_layer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size, embed_dim, model_dim, stride=1, pool_stride=None):
#         super(Convo_attention_layer, self).__init__()
        
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
#         self.relu = nn.ReLU()
#         self.transposed_conv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2)
        
        
#         self.attention = Head(embed_dim=embed_dim, model_dim=model_dim)
#         self.maxpool = nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        
#         # Do we need to flatten ???     TODO 
#         self.flatten = nn.Flatten(2, 3)
#         self.unflatten = nn.Unflatten(2, (-1, -1))
        
#     def forward(self, input):
#         conv_out = self.relu(self.conv(input))
#         trans_out = self.relu(self.transposed_conv(conv_out))

#         batch_size, channels, height, width = trans_out.shape
        
#         reshaped = trans_out.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
#         attention_out, _ = self.attention(reshaped)
#         output = attention_out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
#         pooled = self.maxpool(output)
        
#         return pooled
class Convo_attention_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size, embed_dim, model_dim, stride=1, pool_stride=None):
        super(Convo_attention_layer, self).__init__()

        # Use Conv1d instead of Conv2d
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        #self.relu = nn.ReLU()

        self.transposed_conv = nn.ConvTranspose1d(out_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2)

        self.attention = Head(embed_dim=embed_dim, model_dim=model_dim)

        self.avgpool = nn.AvgPool1d(pool_kernel_size, stride=pool_stride)

    def forward(self, input):
        # input: [B, 4, 100]
        conv_out = self.conv(input)                # [B, out_channels, 100]
        trans_out = self.transposed_conv(conv_out)  # [B, in_channels, 100] = [B, 4, 100]

        batch_size, channels, length = trans_out.shape

        reshaped = trans_out.permute(0, 2, 1)  # [B, 100, 4] for attention
        attention_out, _ = self.attention(reshaped)  # [B, 100, model_dim] or [B, 100, 4]

        output = attention_out.permute(0, 2, 1)  # [B, 4, 100] if model_dim == 4
        pooled = self.avgpool(output)           # If pool_kernel_size=1, shape remains [B, 4, 100]

        return pooled


class N_Convo_attention(nn.Module):
    def __init__(self, config):
        super(N_Convo_attention, self).__init__()
        
        self.layers = nn.ModuleList([
            Convo_attention_layer(
                config.in_channels,
                config.out_channels,
                config.kernel_size,
                config.pool_kernel_size,
                config.embed_dim,  # embed_dim parameter
                config.model_dim,  # model_dim parameter
                config.stride,
                config.pool_stride
            ) for _ in range(config.num_convo_attention_layers)
        ])
    
    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


class conv1d_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pool_kernel_size=2, pool_stride=None, embed_dim=None, model_dim=None):
        super(conv1d_attention, self).__init__()
        
        self.downsample = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=100, stride=100)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.transposed_conv = nn.ConvTranspose1d(out_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2)

        self.attention = Head(embed_dim=embed_dim, model_dim=model_dim)
        self.avgpool = nn.AvgPool1d(pool_kernel_size, stride=pool_stride)

    def forward(self, input):
        input = input.permute(0, 2, 1)  #[B, L, C] -> [B, C, L]
        input = self.downsample(input)  # Downsample the input to  [B, 4, 200]]

        conv_out = self.conv(input)
        trans_out = self.transposed_conv(conv_out)

        batch_size, channels, length = trans_out.shape

        reshaped = trans_out.permute(0, 2, 1).reshape(batch_size, length, channels)
        attention_out, _ = self.attention(reshaped)
        output = attention_out.reshape(batch_size, length, channels).permute(0, 2, 1)

        pooled = self.avgpool(output)

        return pooled