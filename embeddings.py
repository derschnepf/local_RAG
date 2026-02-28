from sentence_transformers import SentenceTransformer
MODELL_NAME = 'all-MiniLM-L6-v2'
print(f"Lade Embedding-Modell '{MODELL_NAME}'... (Das kann beim ersten Mal kurz dauern)")
modell = SentenceTransformer(MODELL_NAME)

def generiere_embeddings(chunks):
    if not chunks:
        return []
    embeddings = modell.encode(chunks, show_progress_bar=True)
    return embeddings