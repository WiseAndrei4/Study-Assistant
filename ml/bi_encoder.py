import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BiEncoder(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", tau: float = 0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tau = tau

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(out.size()).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    def infonce_loss(
            self,
            q_emb: torch.Tensor,  # [B, D]
            pos_emb: torch.Tensor,  # [B, D]
            hard_neg_emb: torch.Tensor  # [B, M, D]
    ) -> torch.Tensor:
        # In-batch logits: [B, B]
        in_batch_logits = torch.matmul(q_emb, pos_emb.T) / self.tau
        # Hard-negative logits: [B, M]
        hard_neg_logits = torch.einsum("bd,bmd->bm", q_emb, hard_neg_emb) / self.tau
        logits = torch.cat([in_batch_logits, hard_neg_logits], dim=1)
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        return F.cross_entropy(logits, labels)
