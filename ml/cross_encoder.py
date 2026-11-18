import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        scores = []
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())
        return scores