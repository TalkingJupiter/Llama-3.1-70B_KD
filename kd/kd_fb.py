import torch
import torch.nn as nn

class LinearProjector(nn.Module):
    def __init__():
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):
        return self.proj(x)

def feature_kd_loss(student_feats, teacher_feats, token_mask=None):
    diff = student_feats - teacher_feats
    if token_mask is not None:
        mask = token_mask.bool().unsqueeze(-1)
        diff = diff[mask]
    return (diff ** 2).mean()