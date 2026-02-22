"""Modular embedding pipelines: SentenceTransformer, BERT-based and Dummy encoders.

Expose a common interface:
    encoder = Encoder(...)
    vectors = encoder.encode(list_of_texts, batch_size=...)

Heavy dependencies (torch, transformers, sentence_transformers) are imported lazily inside classes
so that lightweight smoke runs can use `DummyEncoder` without requiring PyTorch.
"""
from typing import List
import numpy as np
from tqdm import tqdm


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)
        return embeddings


class BertEncoder:
    """BERT-based encoder using HuggingFace transformers.
    Produces mean-pooled token embeddings from last hidden state (or [CLS] if desired).
    """
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None, use_cls: bool = False):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.use_cls = use_cls
        self.dim = self.model.config.hidden_size

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = __import__('torch').sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = __import__('torch').clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        import torch
        all_embs = []
        iterator = tqdm(range(0, len(texts), batch_size), disable=not show_progress)
        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                model_output = self.model(input_ids, attention_mask=attention_mask)
                if self.use_cls:
                    emb = model_output[0][:, 0, :]
                else:
                    emb = self._mean_pooling(model_output, attention_mask)
                emb = emb.cpu().numpy()
                all_embs.append(emb)
        return np.vstack(all_embs)


class DummyEncoder:
    """Lightweight encoder for demo/smoke runs that does not require PyTorch or scikit-learn.

    Behavior:
    - If `sklearn` is available, uses `TfidfVectorizer` (same as before).
    - Otherwise falls back to a simple hashing-TF implementation (pure Python, deterministic).
    """
    def __init__(self, max_dim: int = 768):
        self.dim = max_dim
        self._fitted = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self.vectorizer = TfidfVectorizer(max_features=max_dim)
            self._use_sklearn = True
        except Exception:
            # fallback: simple hashing-based vectorizer
            self.vectorizer = None
            self._use_sklearn = False

    def _hash_vectorize(self, texts: List[str]) -> np.ndarray:
        # simple hashing TF: token -> bucket via Python hash
        arr = np.zeros((len(texts), self.dim), dtype='float32')
        for i, txt in enumerate(texts):
            for tok in txt.split():
                idx = (hash(tok) % self.dim)
                arr[i, idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(arr[i])
            if norm > 0:
                arr[i] /= norm
        return arr

    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        if self._use_sklearn:
            if not self._fitted:
                X = self.vectorizer.fit_transform(texts)
                self._fitted = True
            else:
                X = self.vectorizer.transform(texts)
            arr = X.toarray().astype('float32')
            if arr.shape[1] < self.dim:
                pad = np.zeros((arr.shape[0], self.dim - arr.shape[1]), dtype='float32')
                arr = np.hstack([arr, pad])
            elif arr.shape[1] > self.dim:
                arr = arr[:, : self.dim]
            return arr
        else:
            return self._hash_vectorize(texts)

"""Modular embedding pipelines: SentenceTransformer, BERT-based and Dummy encoders.

Expose a common interface:
    encoder = Encoder(...)
    vectors = encoder.encode(list_of_texts, batch_size=...)

Heavy dependencies (torch, transformers, sentence_transformers) are imported lazily inside classes
so that lightweight smoke runs can use `DummyEncoder` without requiring PyTorch.
"""
from typing import List
import numpy as np
from tqdm import tqdm


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)
        return embeddings


class BertEncoder:
    """BERT-based encoder using HuggingFace transformers.
    Produces mean-pooled token embeddings from last hidden state (or [CLS] if desired).
    """
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None, use_cls: bool = False):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.use_cls = use_cls
        self.dim = self.model.config.hidden_size

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = __import__('torch').sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = __import__('torch').clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        import torch
        all_embs = []
        iterator = tqdm(range(0, len(texts), batch_size), disable=not show_progress)
        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                model_output = self.model(input_ids, attention_mask=attention_mask)
                if self.use_cls:
                    emb = model_output[0][:, 0, :]
                else:
                    emb = self._mean_pooling(model_output, attention_mask)
                emb = emb.cpu().numpy()
                all_embs.append(emb)
        return np.vstack(all_embs)


class DummyEncoder:
    """Lightweight encoder for demo/smoke runs that does not require PyTorch or scikit-learn.

    Behavior:
    - If `sklearn` is available, uses `TfidfVectorizer` (same as before).
    - Otherwise falls back to a simple hashing-TF implementation (pure Python, deterministic).
    """
    def __init__(self, max_dim: int = 768):
        self.dim = max_dim
        self._fitted = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self.vectorizer = TfidfVectorizer(max_features=max_dim)
            self._use_sklearn = True
        except Exception:
            # fallback: simple hashing-based vectorizer
            self.vectorizer = None
            self._use_sklearn = False

    def _hash_vectorize(self, texts: List[str]) -> np.ndarray:
        # simple hashing TF: token -> bucket via Python hash
        arr = np.zeros((len(texts), self.dim), dtype='float32')
        for i, txt in enumerate(texts):
            for tok in txt.split():
                idx = (hash(tok) % self.dim)
                arr[i, idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(arr[i])
            if norm > 0:
                arr[i] /= norm
        return arr

    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        if self._use_sklearn:
            if not self._fitted:
                X = self.vectorizer.fit_transform(texts)
                self._fitted = True
            else:
                X = self.vectorizer.transform(texts)
            arr = X.toarray().astype('float32')
            if arr.shape[1] < self.dim:
                pad = np.zeros((arr.shape[0], self.dim - arr.shape[1]), dtype='float32')
                arr = np.hstack([arr, pad])
            elif arr.shape[1] > self.dim:
                arr = arr[:, : self.dim]
            return arr
        else:
            return self._hash_vectorize(texts)
