Author: Belma Đelilovic
# Evaluacija semantičke pretrage u Weaviate i Pinecone

Opis: Eksperimentalni pipeline za upoređivanje performansi i relevantnosti semantičke pretrage koristeći Python, BERT i SentenceTransformer.

Struktura projekta:
```
project/
│── data/
│── embeddings/
│── weaviate/
│── pinecone/
│── ingestion/
│── evaluation/
│── experiments/
│── plots/
│── main.py
│── requirements.txt
```

Vodič za pokretanje:
1. Instalirajte zavisnosti: `pip install -r requirements.txt`
2. Pripremite podatke: `python main.py --action prepare_data --n_docs 100000`
3. Pokrenite eksperimente (potrebne su Pinecone/Weaviate konfiguracije): `python main.py --action run_all`

Reproducible smoke test (recommended for quick demo):

1. Create a virtual environment and install packages:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Run the automated smoke script which prepares 1,000 documents and runs a short experiment for both models:

```powershell
.\run_smoke.ps1
```

3. Outputs: experiment results and CSV summaries are saved under `experiments/results/`.

Files added for reproducibility:
- `.env.example` — example environment variables for Pinecone/Weaviate
- `run_smoke.ps1` — automated smoke-run script
- `README_RUN_LARGE.md` — instructions for large-scale runs and recommendations

If you want, I can package the repository (excluding `.venv`) into a ZIP ready to send to your professor.

