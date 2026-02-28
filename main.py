import os
from ingestion import lade_pdf_text
from chunking import erstelle_chunks
from embeddings import generiere_embeddings

if __name__ == "__main__":
    # Achte darauf, dass der Dateiname exakt mit deiner PDF übereinstimmt!
    pdf_pfad = "data_science_buch.pdf" 

    print("--- Starte lokales RAG-System Pipeline ---")

    if not os.path.exists(pdf_pfad):
        print(f"FEHLER: Die Datei '{pdf_pfad}' existiert nicht in diesem Ordner.")
    else:
        # Schritt 1 & 2: Laden und Chunking
        print("\n[1/3] Lese PDF ein...")
        gesamter_text = lade_pdf_text(pdf_pfad)
        
        if gesamter_text:
            print("\n[2/3] Erstelle Text-Chunks...")
            chunks = erstelle_chunks(gesamter_text, chunk_groesse=1000, overlap=200)
            print(f"      {len(chunks)} Chunks erstellt.")
            
            # Schritt 3: Embeddings generieren
            print("\n[3/3] Generiere Embeddings (Vektoren)...")
            vektoren = generiere_embeddings(chunks)
            
            # Kontrolle: Wie sehen unsere Daten jetzt aus?
            print("\n=== ERFOLG! Pipeline abgeschlossen ===")
            print(f"Anzahl der erstellten Vektoren: {len(vektoren)}")
            print(f"Dimensionen pro Vektor: {len(vektoren[0])} Zahlen")
            print("======================================")
            
        else:
            print("FEHLER: Konnte keinen Text extrahieren.")