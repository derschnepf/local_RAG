def erstelle_chunks(text, chunk_groesse=1000, overlap=200):
    """
    Teilt einen langen Text in kleinere Abschnitte (Chunks) mit Überlappung auf.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_laenge = len(text)

    while start < text_laenge:
        ende = start + chunk_groesse
        chunk = text[start:ende]
        chunks.append(chunk)
        start += chunk_groesse - overlap

    return chunks