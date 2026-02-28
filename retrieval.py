import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

def lade_datenbank(speicherort='./faiss_db'):
    index_pfad = os.path.join(speicherort, 'vektoren.index')
    chunks_pfad = os.path.join(speicherort, 'chunks.pkl')
    if not os.path.exists(index_pfad) or not os.path.exists(chunks_pfad):
        return (None, None)
    index = faiss.read_index(index_pfad)
    with open(chunks_pfad, 'rb') as f:
        chunks = pickle.load(f)
    return (index, chunks)

def suche_relevante_chunks(frage, index, chunks, modell, top_k=3):
    frage_vektor = modell.encode([frage])
    frage_vektor_np = np.array(frage_vektor).astype('float32')
    abstaende, indizes = index.search(frage_vektor_np, top_k)
    gefundene_chunks = [chunks[i] for i in indizes[0]]
    return gefundene_chunks

def generiere_antwort(frage, kontext_chunks):
    kontext_text = '\n\n'.join(kontext_chunks)
    prompt = f'Du bist ein intelligenter Data-Science-Assistent. \n    Beantworte die folgende Frage AUSSCHLIESSLICH basierend auf dem bereitgestellten KONTEXT aus einem Buch. \n    Wenn die Antwort nicht im Kontext steht, sage höflich, dass du die Antwort im Buch nicht finden kannst.\n    Antworte auf Deutsch.\n\n    KONTEXT:\n    {kontext_text}\n\n    FRAGE: \n    {frage}\n    '
    response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']
if __name__ == '__main__':
    print('Starte RAG-System... Lade Datenbank und Embedding-Modell.')
    faiss_index, alle_chunks = lade_datenbank()
    if faiss_index is None:
        print("FEHLER: Datenbank 'faiss_db' nicht gefunden.")
    else:
        embedding_modell = SentenceTransformer('all-MiniLM-L6-v2')
        print('\nRAG-SYSTEM BEREIT')
        print("Tippe 'exit' ein, um das Programm zu beenden.")
        while True:
            frage = input('\nStelle eine Frage zu deinem Data Science Buch: ')
            if frage.lower() == 'exit':
                print('Bis zum nächsten Mal!')
                break
            print('Status: Suche')
            ergebnisse = suche_relevante_chunks(frage, faiss_index, alle_chunks, embedding_modell, top_k=3)
            print('Status: Generiere Antwort')
            antwort = generiere_antwort(frage, ergebnisse)
            print('\n' + '=' * 50)
            print('Antwort:')
            print('=' * 50)
            print(antwort)
            print('\n' + '-' * 50)
            print('Quellen:')
            for i, text in enumerate(ergebnisse):
                print(f"[{i + 1}] {text[:100].replace('\n', ' ')}...")
            print('-' * 50)