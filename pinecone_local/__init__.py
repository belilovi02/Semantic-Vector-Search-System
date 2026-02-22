"""Local wrapper package exposing Pinecone helpers under a different name.

Use `from pinecone_local import client as pinecone_client_module` in code to avoid name collision
with the installed `pinecone` package.
"""

__all__ = ["client"]
