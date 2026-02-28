from sentence_transformers import SentenceTransformer

# Wir laden das Modell auf Dateiebene, damit es nur EINMAL in den Speicher geladen wird.
# Beim allerersten Ausführen lädt Python das Modell (ca. 80 MB) aus dem Internet herunter.
MODELL_NAME = 'all-MiniLM-L6-v2'
print(f"Lade Embedding-Modell '{MODELL_NAME}'... (Das kann beim ersten Mal kurz dauern)")
modell = SentenceTransformer(MODELL_NAME)

def generiere_embeddings(chunks):
    """
    Nimmt eine Liste von Text-Chunks und wandelt sie in Vektoren (Embeddings) um.
    """
    if not chunks:
        return []
    
    # Die .encode() Funktion macht die ganze harte KI-Arbeit für uns.
    # show_progress_bar=True zeigt uns einen schönen Ladebalken im Terminal an.
    embeddings = modell.encode(chunks, show_progress_bar=True)
    
    return embeddings