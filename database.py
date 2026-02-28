import faiss
import numpy as np
import pickle
import os

def speichere_in_db(chunks, embeddings, speicherort='./faiss_db'):
    if len(chunks) == 0 or len(embeddings) == 0:
        print('Keine Daten zum Speichern vorhanden.')
        return None
    if not os.path.exists(speicherort):
        os.makedirs(speicherort)
    vektoren_np = np.array(embeddings).astype('float32')
    dimensionen = len(vektoren_np[0])
    index = faiss.IndexFlatL2(dimensionen)
    index.add(vektoren_np)
    index_pfad = os.path.join(speicherort, 'vektoren.index')
    faiss.write_index(index, index_pfad)
    chunks_pfad = os.path.join(speicherort, 'chunks.pkl')
    with open(chunks_pfad, 'wb') as f:
        pickle.dump(chunks, f)
    return index