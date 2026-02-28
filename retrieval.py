import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama  # NEU: Unsere Brücke zu Llama 3.2

# ==========================================
# WERKZEUGE FÜR DIE SUCHE
# ==========================================
def lade_datenbank(speicherort="./faiss_db"):
    """Lädt den FAISS-Index und die Text-Chunks von der Festplatte."""
    index_pfad = os.path.join(speicherort, "vektoren.index")
    chunks_pfad = os.path.join(speicherort, "chunks.pkl")
    
    if not os.path.exists(index_pfad) or not os.path.exists(chunks_pfad):
        return None, None
        
    index = faiss.read_index(index_pfad)
    with open(chunks_pfad, "rb") as f:
        chunks = pickle.load(f)
        
    return index, chunks

def suche_relevante_chunks(frage, index, chunks, modell, top_k=3):
    """Sucht die ähnlichsten Textabschnitte in der FAISS-Datenbank."""
    frage_vektor = modell.encode([frage])
    frage_vektor_np = np.array(frage_vektor).astype('float32')
    abstaende, indizes = index.search(frage_vektor_np, top_k)
    gefundene_chunks = [chunks[i] for i in indizes[0]]
    return gefundene_chunks

# ==========================================
# NEU: WERKZEUG FÜR DIE ANTWORT-GENERIERUNG
# ==========================================
def generiere_antwort(frage, kontext_chunks):
    """
    Baut einen Prompt aus der Frage und dem Buch-Kontext und lässt 
    Llama 3.2 die finale Antwort generieren.
    """
    # 1. Wir verbinden die 3 gefundenen Chunks zu einem langen Text
    kontext_text = "\n\n".join(kontext_chunks)
    
    # 2. Das Herzstück: Das Prompt-Engineering! 
    # Wir geben der KI strenge Regeln, damit sie nicht halluziniert.
    prompt = f"""Du bist ein intelligenter Data-Science-Assistent. 
    Beantworte die folgende Frage AUSSCHLIESSLICH basierend auf dem bereitgestellten KONTEXT aus einem Buch. 
    Wenn die Antwort nicht im Kontext steht, sage höflich, dass du die Antwort im Buch nicht finden kannst.
    Antworte auf Deutsch.

    KONTEXT:
    {kontext_text}

    FRAGE: 
    {frage}
    """

    # 3. Wir schicken den Prompt an unser lokales Ollama-Modell
    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])
    
    # 4. Wir extrahieren nur den reinen Text der Antwort
    return response['message']['content']

# ==========================================
# INTERAKTIVES HAUPTPROGRAMM
# ==========================================
if __name__ == "__main__":
    print("Starte RAG-System... Lade Datenbank und Embedding-Modell.")
    
    faiss_index, alle_chunks = lade_datenbank()
    
    if faiss_index is None:
        print("FEHLER: Datenbank 'faiss_db' nicht gefunden.")
    else:
        embedding_modell = SentenceTransformer('all-MiniLM-L6-v2')
        print("\n=== RAG-SYSTEM BEREIT! ===")
        print("Tippe 'exit' ein, um das Programm zu beenden.")
        
        while True:
            frage = input("\nStelle eine Frage zu deinem Data Science Buch: ")
            
            if frage.lower() == 'exit':
                print("Bis zum nächsten Mal!")
                break
                
            print("🔍 Durchsuche das Buch...")
            ergebnisse = suche_relevante_chunks(frage, faiss_index, alle_chunks, embedding_modell, top_k=3)
            
            print("🧠 Llama 3.2 formuliert die Antwort... (Das kann ein paar Sekunden dauern)")
            antwort = generiere_antwort(frage, ergebnisse)
            
            # Ausgabe der KI-Antwort
            print("\n" + "="*50)
            print("🤖 ANTWORT:")
            print("="*50)
            print(antwort)
            
            # Transparenz: Zeigen, woher die KI ihr Wissen hat
            print("\n" + "-"*50)
            print("📚 Quellen (Die verwendeten Buch-Abschnitte):")
            for i, text in enumerate(ergebnisse):
                # Wir zeigen nur die ersten 100 Zeichen zur Orientierung
                print(f"[{i+1}] {text[:100].replace('\n', ' ')}...") 
            print("-" * 50)