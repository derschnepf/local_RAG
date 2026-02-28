import fitz  # Pip install PyMuPDF

def lade_pdf_text(dateipfad):
    """
    Öffnet eine PDF-Datei mit PyMuPDF, liest den Text aller Seiten und gibt ihn als String zurück.
    """
    text = ""
    try:
        # Dokument öffnen
        dokument = fitz.open(dateipfad)
        
        # Über alle Seiten iterieren
        for seite in dokument:
            extrahierter_text = seite.get_text()
            if extrahierter_text:
                text += extrahierter_text + "\n"
                
        dokument.close()
        return text
        
    except Exception as e:
        print(f"Fehler beim Lesen der PDF: {e}")
        return None