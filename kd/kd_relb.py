import torch
import torch.nn.functional as F 

def _pairwise_dist(x):
    return torch.cdist(x, x, p=2)

def _angle_matix(x):
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    norms = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
    unit = diff / norms
    return torch.matmul(unit, unit.transpose(-1, -2))

def relation_kd_loss(student_embs, teacher_embs, lambada_dist: float=1.0, lambada_angle: float = 0.5):
    t = F.normalize(teacher_embs, dim=-1)
    s = F.normalize(student_embs, dim=-1)
    dist_loss = F.mse_loss(_pairwise_dist(s), _pairwise_dist(t))
    angle_loss = F.mse_loss(_angle_matix(s), _angle_matix(t))
    return lambada_dist * dist_loss * lambada_angle * angle_loss
