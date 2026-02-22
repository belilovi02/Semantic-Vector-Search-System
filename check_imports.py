import importlib
import sys

packages = [
    'datasets',
    'sentence_transformers',
    'transformers',
    'torch',
    'pandas',
    'weaviate',
    'pinecone',
]

success = True
for p in packages:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, '__version__', 'unknown')
        print(f"{p}: OK (version={ver})")
    except Exception as e:
        print(f"{p}: FAIL -> {e}")
        success = False

if not success:
    sys.exit(2)

print('All checked imports attempted.')
