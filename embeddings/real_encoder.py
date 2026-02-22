"""Real encoder wrapper using sentence-transformers for embeddings.

Usage:
    from embeddings.real_encoder import RealEncoder
    enc = RealEncoder(model_name='all-MiniLM-L6-v2')
    vecs = enc.encode(["hello world"], batch_size=8)
"""
from typing import List
import numpy as np

class RealEncoder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        # Try sentence-transformers first (best embeddings for semantic search)
        try:
            from sentence_transformers import SentenceTransformer
            self.backend = 'sentence_transformers'
        except Exception as e:
            # fallback to HF transformers-based mean-pooling encoder
            self.backend = 'transformers'
            self._st_error = e

        # Use CUDA if available unless user specified otherwise
        try:
            import torch
            _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            _device = 'cpu'
        self.device = device or _device
        self.model_name = model_name

        if self.backend == 'sentence_transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            # transformers-based encoder (mean pooling over last hidden state)
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
            except Exception as e:
                raise RuntimeError('transformers must be installed for HF fallback encoder') from e
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name)
            try:
                self.model.to(self.device)
            except Exception:
                pass

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if self.backend == 'sentence_transformers':
            embs = self.model.encode(texts, batch_size=batch_size, show_progress=show_progress, convert_to_numpy=True)
            return np.asarray(embs)
        else:
            import torch
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.model.device)
                attention_mask = encoded['attention_mask'].to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self._mean_pooling(outputs, attention_mask)
                embs = pooled.cpu().numpy()
                all_embs.append(embs)
            return np.vstack(all_embs)

