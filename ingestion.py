import fitz

def lade_pdf_text(dateipfad):
    text = ''
    try:
        dokument = fitz.open(dateipfad)
        for seite in dokument:
            extrahierter_text = seite.get_text()
            if extrahierter_text:
                text += extrahierter_text + '\n'
        dokument.close()
        return text
    except Exception as e:
        print(f'Fehler beim Lesen der PDF: {e}')
        return None