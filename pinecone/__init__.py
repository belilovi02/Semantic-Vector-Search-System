"""Local pinecone package wrapper for project-specific helpers.

This file makes the `pinecone` folder a package so `import pinecone.client`
resolves to the local `pinecone/client.py` module instead of the installed package.
"""

__all__ = ["client"]
