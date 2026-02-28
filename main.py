import os
from ingestion import lade_pdf_text
from chunking import erstelle_chunks
from embeddings import generiere_embeddings
from database import speichere_in_db
if __name__ == '__main__':
    pdf_pfad = 'data_science_buch.pdf'
    print('--- Starte lokales RAG-System Pipeline ---')
    if not os.path.exists(pdf_pfad):
        print(f"FEHLER: Die Datei '{pdf_pfad}' existiert nicht.")
    else:
        print('\n[1/4] Lese PDF ein...')
        gesamter_text = lade_pdf_text(pdf_pfad)
        if gesamter_text:
            print('\n[2/4] Erstelle Text-Chunks...')
            chunks = erstelle_chunks(gesamter_text, chunk_groesse=1000, overlap=200)
            print(f'      {len(chunks)} Chunks erstellt.')
            print('\n[3/4] Generiere Embeddings (Vektoren)...')
            vektoren = generiere_embeddings(chunks)
            print('\n[4/4] Speichere Daten in ChromaDB...')
            speichere_in_db(chunks, vektoren)
            print('\n=== ERFOLG! Pipeline abgeschlossen ===')
            print("Die Vektoren liegen jetzt sicher auf deiner Festplatte im Ordner 'chroma_db'!")
            print('======================================')
        else:
            print('FEHLER: Konnte keinen Text extrahieren.')