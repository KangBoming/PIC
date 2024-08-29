import torch
import torch.nn as nn
import torch.nn.functional as F

class PIC(nn.Module):
    def __init__(self, input_shape:int, output_shape:int, hidden_units,attn_drop, linear_drop,device) -> None:
        super().__init__()
        self.device = device
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_shape, num_heads=1, dropout=attn_drop, batch_first=True)
        self.layerNorm=nn.LayerNorm(input_shape)
        self.generator = nn.Sequential(
            nn.BatchNorm1d(input_shape),
            nn.Linear(in_features=input_shape, out_features=hidden_units*2),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.BatchNorm1d(hidden_units*2),
            nn.Linear(in_features=hidden_units*2,out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
            )
    def forward(self, x, start_padding_indices, get_attention=False):
        batch_size, seq_len, input_shape = x.shape
        key_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(self.device)
        for i, start_padding_idx in enumerate(start_padding_indices):
            if start_padding_idx != -1:
                key_padding_mask[i, start_padding_idx:] = True
        output, attention_weights = self.multihead_attention(x, x, x, key_padding_mask=key_padding_mask)
        output=x+self.layerNorm(output) 
        pooled_feature = torch.zeros(batch_size, input_shape).to(self.device)
        for i, start_padding_idx in enumerate(start_padding_indices):
            if start_padding_idx != -1:
                pooled_feature[i] = torch.mean(output[i,:start_padding_idx, :], dim=0).to(self.device)
            else:
                pooled_feature[i] = torch.mean(output[i,:,:], dim=0).to(self.device)
        if get_attention:
            return self.generator(pooled_feature), attention_weights
        else:
            return self.generator(pooled_feature)
        

        