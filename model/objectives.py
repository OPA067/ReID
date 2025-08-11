import torch.nn.functional as F

def compute_InfoNCE_per(scores, logit_scale):

    logits = logit_scale * scores
    logits_t = logits.t()

    p1 = F.softmax(logits, dim=1)
    p2 = F.softmax(logits_t, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2
    return loss

def compute_loss(tar_feats, can_feats, logit_scale=50):

    tar_feats = tar_feats / tar_feats.norm(dim=-1, keepdim=True)
    can_feats = can_feats / can_feats.norm(dim=-1, keepdim=True)
    scores = tar_feats @ can_feats.t()

    loss_tar_can = compute_InfoNCE_per(scores, logit_scale)
    loss_tar_can = loss_tar_can.sum()

    return loss_tar_can



