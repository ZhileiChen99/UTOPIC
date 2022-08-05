import torch
import torch.nn as nn
from models.geo_attention import GeometryTransformer


class TransformerModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_head, blocks):
        super(TransformerModule, self).__init__()

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = GeometryTransformer(blocks, hidden_dim, num_head, dropout=None, activation_fn='relu')
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, src_feats, ref_feats, src_geo, ref_geo, src_masks=None, ref_masks=None):
        src_feats = self.in_proj(src_feats)
        ref_feats = self.in_proj(ref_feats)

        src_feats, ref_feats = self.transformer(
            src_feats, ref_feats, src_geo, ref_geo, masks0=src_masks, masks1=ref_masks
        )
        src_feats = self.out_proj(src_feats)
        ref_feats = self.out_proj(ref_feats)
        return src_feats, ref_feats
