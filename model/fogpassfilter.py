import torch
import torch.nn as nn
import torch.nn.functional as F

class FogPassFilter_conv1(nn.Module):
    def __init__(self, inputsize):
        super(FogPassFilter_conv1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 16)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class FogPassFilter_res1(nn.Module):
    def __init__(self, inputsize):
        super(FogPassFilter_res1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize//8)
        self.output = nn.Linear(inputsize//8, 8)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class FogPassFilterLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Compute pairwise cosine distances
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cosine_dist = 1 - torch.mm(norm_embeddings, norm_embeddings.t())

        # Create mask for pairs from same domain (1) and different domains (0)
        labels_mat = labels.expand(len(labels), len(labels))
        same_domain_mask = (labels_mat == labels_mat.t()).float()

        # Compute positive pair loss: (1 - I(a,b))[m - d(F^a, F^b)]_+^2
        pos_pair_loss = (1 - same_domain_mask) * torch.pow(
            torch.clamp(self.margin - cosine_dist, min=0), 2
        )

        # Compute negative pair loss: I(a,b)[d(F^a, F^b) - m]_+^2
        neg_pair_loss = same_domain_mask * torch.pow(
            torch.clamp(cosine_dist - self.margin, min=0), 2
        )

        # Sum both terms and remove diagonal elements
        mask = 1 - torch.eye(len(labels), device=labels.device)
        total_loss = (pos_pair_loss + neg_pair_loss) * mask

        # Return mean over all valid pairs
        return total_loss.sum() / (mask.sum() + 1e-8)

class PairedFogPassFilterLoss(nn.Module):
    """
    Loss for CW-SF matching pairs only.
    Two variants:
      - 'mse'   : L = mean ||emb_cw - emb_sf||^2
      - 'cos'   : L = mean (1 - cos_sim)^2  (cosine-distance squared)
    emb_cw / emb_sf should be shape (B, D).
    """
    def __init__(self, variant='mse', normalize=True):
        """
        variant: 'mse' or 'cos'
        normalize: whether to L2-normalize embeddings before computing loss
        """
        super().__init__()
        assert variant in ('mse', 'cos')
        self.variant = variant
        self.normalize = normalize

    def forward(self, emb_cw, emb_sf):
        """
        emb_cw, emb_sf: tensors (B, D)
        returns scalar loss
        """
        # shapes check
        if emb_cw.dim() != 2 or emb_sf.dim() != 2:
            raise ValueError("emb_cw / emb_sf must be (B, D)")
        if emb_cw.size(0) != emb_sf.size(0):
            raise ValueError("CW and SF must have same batch size and be aligned (cw_i matches sf_i).")

        if self.normalize:
            emb_cw = F.normalize(emb_cw, p=2, dim=1)
            emb_sf = F.normalize(emb_sf, p=2, dim=1)

        if self.variant == 'mse':
            # squared L2 per-pair, averaged
            per_pair = (emb_cw - emb_sf).pow(2).sum(dim=1)   # (B,)
            loss = per_pair.mean()
            return loss
        else:  # 'cos'
            cos = (emb_cw * emb_sf).sum(dim=1)              # cosine similarity per pair
            dist = 0.1 - cos                                # cosine distance
            loss = (dist.pow(2)).mean()
            return loss