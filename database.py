import faiss
import numpy as np
import pickle
import os

def speichere_in_db(chunks, embeddings, speicherort="./faiss_db"):
    """
    Speichert Vektoren in einem FAISS-Index und die dazugehörigen Text-Chunks 
    in einer separaten Datei.
    """
    # HIER IST DER FIX: Wir nutzen len() anstelle von 'not'
    if len(chunks) == 0 or len(embeddings) == 0:
        print("Keine Daten zum Speichern vorhanden.")
        return None

    # 1. Speicherordner erstellen, falls er nicht existiert
    if not os.path.exists(speicherort):
        os.makedirs(speicherort)

    # 2. FAISS braucht die Daten in einem ganz bestimmten Zahlenformat (numpy float32)
    vektoren_np = np.array(embeddings).astype('float32')
    dimensionen = len(vektoren_np[0])

    # 3. Den FAISS-Index erstellen 
    index = faiss.IndexFlatL2(dimensionen)
    
    # 4. Die Vektoren in den Index laden
    index.add(vektoren_np)
    
    # 5. Den Vektor-Index auf der Festplatte speichern
    index_pfad = os.path.join(speicherort, "vektoren.index")
    faiss.write_index(index, index_pfad)

    # 6. Die originalen Texte (Chunks) speichern (mit 'pickle')
    chunks_pfad = os.path.join(speicherort, "chunks.pkl")
    with open(chunks_pfad, "wb") as f:
        pickle.dump(chunks, f)

    return index