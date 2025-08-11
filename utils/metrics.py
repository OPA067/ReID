from prettytable import PrettyTable
import torch
import numpy as np
import logging
from tqdm import tqdm

def metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['Rsum'] = metrics['R1'] + metrics['R5'] + metrics['R10']
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MdR"] = metrics['MR']
    metrics["MnR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def get_metrics(similarity, n_):
    p2p_metrics = metrics(similarity)
    r1, r5, r10, rsum, mdr, mnr = p2p_metrics['R1'], p2p_metrics['R5'], p2p_metrics['R10'], p2p_metrics['Rsum'], p2p_metrics['MdR'], p2p_metrics['MnR']
    return [n_, r1, r5, r10, rsum, mdr, mnr]

class Evaluator():
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.logger = logging.getLogger("ReID")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        id_list, tar_feat_list, can_feat_list = [], [], []

        for n_iter, batch in tqdm(enumerate(self.test_loader)):
            id = batch['id']
            tar_img = batch['tar_img'].to(device)
            can_img = batch['can_img'].to(device)
            with torch.no_grad():
                tar_feat, can_feat = model.encode_image(tar_img), model.encode_image(can_img)
                tar_feat = tar_feat / tar_feat.norm(dim=-1, keepdim=True)
                can_feat = can_feat / can_feat.norm(dim=-1, keepdim=True)

            id_list.extend(id)
            tar_feat_list.append(tar_feat)
            can_feat_list.append(can_feat)

        id_list = np.array(id_list)
        tar_feat_list = torch.cat(tar_feat_list, 0)
        can_feat_list = torch.cat(can_feat_list, 0)

        return id_list, tar_feat_list.cpu(), can_feat_list.cpu()

    def eval(self, model):
        _, tar_feat_list, can_feat_list = self._compute_embedding(model)

        tar_feat_list = tar_feat_list / tar_feat_list.norm(dim=-1, keepdim=True)
        can_feat_list = can_feat_list / can_feat_list.norm(dim=-1, keepdim=True)
        sims = tar_feat_list @ can_feat_list.t()

        sims_dict = {
            'sims': sims, # [cls]
        }
        table = PrettyTable(["task", "R1", "R5", "R10", "RSum", "MdR", "MnR"])
        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, f'{key}')
            table.add_row(rs)

        table.custom_format["R1"] = lambda f, v: f"{v:.1f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.1f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.1f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.1f}"
        table.custom_format["MnR"] = lambda f, v: f"{v:.1f}"
        table.custom_format["MdR"] = lambda f, v: f"{v:.1f}"
        self.logger.info('\n' + str(table))

        return rs[4] # RSum
