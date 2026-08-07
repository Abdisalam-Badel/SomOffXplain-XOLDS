import torch
import torch.nn.functional as F

class ContrastiveRationaleLoss:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def class_contrastive(self, span_embs, labels):
        span_embs = F.normalize(span_embs, dim=-1)
        device = span_embs.device
        batch_size = span_embs.size(0)
        sim_matrix = torch.matmul(span_embs, span_embs.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size, device=device)
        mask = mask - mask_self

        num_positives = (mask.sum(1) > 0).float().sum()
        if num_positives == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss


 

