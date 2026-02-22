"""Configuration and defaults for experiments."""
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Embedding model defaults
DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BERT_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"  # fallback; fine-tune as needed

# Ingestion defaults
DEFAULT_BATCH_SIZE = 256

# Experiment sizes
DEFAULT_SIZES = [10000, 50000, 100000]
