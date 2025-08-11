from model import objectives
import torch
import torch.nn as nn
from .clip_model import build_CLIP_from_openai_pretrained
from .cluster import CTM, TCBlock
from .mha import MHAModel_v2

class ReID(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.loss_type = 'InfoNCE'
        self.embed_dim = base_cfg['embed_dim']
        self.agg_model = "None" # ["None", "mlp", "mha", ptm"]

        # None

        # mlp
        if self.agg_model == "mlp":
            self.patch_feat_w = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(inplace=True), nn.Linear(self.embed_dim * 2, 1), )

        # mha
        elif self.agg_model == "mha":
            self.MHA = MHAModel_v2(embed_dim=self.embed_dim)

        # ptm
        elif self.agg_model == "ptm":
            sr = [0.5, 0.5, 0.5]
            self.i_pcm_p_1 = CTM(sample_ratio=sr[0], embed_dim=self.embed_dim, dim_out=self.embed_dim, k=3)
            self.i_att_block_p_1 = TCBlock(dim=self.embed_dim, num_heads=8)
            self.i_pcm_p_2 = CTM(sample_ratio=sr[1], embed_dim=self.embed_dim, dim_out=self.embed_dim, k=3)
            self.i_att_block_p_2 = TCBlock(dim=self.embed_dim, num_heads=8)
            self.i_pcm_p_3 = CTM(sample_ratio=sr[2], embed_dim=self.embed_dim, dim_out=self.embed_dim, k=3)
            self.i_att_block_p_3 = TCBlock(dim=self.embed_dim, num_heads=8)
            self.patch_feat_w = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(inplace=True), nn.Linear(self.embed_dim * 2, 1), )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]

    def get_patch(self, p_feat):
        p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
        p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
        p_mask = p_feat.new_ones(p_feat.size(0), p_feat.size(1))
        p_token_dict = {'x': p_feat,
                        'token_num': p_feat.size(1),
                        'idx_token': p_idx_token,
                        'agg_weight': p_agg_weight,
                        'mask': p_mask.detach()}
        p_token_dict = self.i_att_block_p_1(self.i_pcm_p_1(p_token_dict))
        p_token_dict = self.i_att_block_p_2(self.i_pcm_p_2(p_token_dict))
        p_token_dict = self.i_att_block_p_3(self.i_pcm_p_3(p_token_dict))
        p_feat = p_token_dict['x']
        p_feat_w = torch.softmax(self.patch_feat_w(p_feat).squeeze(dim=-1), dim=-1)
        p_feat = torch.einsum("bpd,bp->bd", [p_feat, p_feat_w])
        return p_feat
    
    def encode_image(self, image):
        # this module is used by testing, edit follow forward...
        feat = self.base_model.encode_image(image)
        return feat

    def forward(self, tar_img, can_img):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        # [cls]
        tar_feat, can_feat = self.base_model(tar_img, can_img)
        loss = objectives.compute_loss(tar_feat, can_feat, logit_scale=self.logit_scale)
        ret.update({'loss': loss})

        """       
        # [cls] + [patch]
        tar_cls_feat, tar_patch_feat, can_cls_patch, can_patch_feat = self.base_model(tar_img, can_img)
        if self.agg_model == "mlp":
            loss_cls = objectives.compute_loss(tar_cls_feat, can_cls_patch, logit_scale=self.logit_scale)
            tar_patch_feat_w = torch.softmax(self.patch_feat_w(tar_patch_feat).squeeze(dim=-1), dim=-1)
            tar_patch_feat = torch.einsum("bpd,bp->bd", [tar_patch_feat, tar_patch_feat_w])
            can_patch_feat_w = torch.softmax(self.patch_feat_w(can_patch_feat).squeeze(dim=-1), dim=-1)
            can_patch_feat = torch.einsum("bpd,bp->bd", [can_patch_feat, can_patch_feat_w])
            loss_patch = objectives.compute_loss(tar_patch_feat, can_patch_feat, logit_scale=self.logit_scale)
            ret.update({'loss': loss_cls + loss_patch})
        elif self.agg_model == "mha":
            loss_cls = objectives.compute_loss(tar_cls_feat, can_cls_patch, logit_scale=self.logit_scale)
            tar_patch_feat = self.MHA(tar_patch_feat)
            can_patch_feat = self.MHA(can_patch_feat)
            loss_patch = objectives.compute_loss(tar_patch_feat, can_patch_feat, logit_scale=self.logit_scale)
            ret.update({'loss': loss_cls + loss_patch})
        elif self.agg_model == "ptm":
            loss_cls = objectives.compute_loss(tar_cls_feat, can_cls_patch, logit_scale=self.logit_scale)
            tar_patch_feat = self.get_patch(tar_patch_feat)
            can_patch_feat = self.get_patch(can_patch_feat)
            loss_patch = objectives.compute_loss(tar_patch_feat, can_patch_feat, logit_scale=self.logit_scale)
            ret.update({'loss': loss_cls + loss_patch})
        """

        return ret

def build_model(args):
    # init model
    model = ReID(args)
    return model
