import os
import shutil

def mou_arxius_a_indian_food(directori):
    """
    Mou tots els fitxers dins de subcarpetes a una carpeta
    'Indian Food Generalitzat' dins el directori actual i elimina subcarpetes buides.
    """
    carpeta_destinacio = os.path.join(directori, "Indian Food Generalitzat")
    os.makedirs(carpeta_destinacio, exist_ok=True)

    # Recorrem el directori complet de manera inversa (bottom-up)
    for arrel, dirs, fitxers in os.walk(directori, topdown=False):
        # Evitem moure fitxers que ja estiguin a la carpeta de destinació
        if arrel == carpeta_destinacio:
            continue

        # Mou fitxers a la carpeta "Indian Food Generalitzat"
        for fitxer in fitxers:
            cami_actual = os.path.join(arrel, fitxer)
            cami_destinacio = os.path.join(carpeta_destinacio, fitxer)

            # Evita sobreescriure fitxers
            if os.path.exists(cami_destinacio):
                nom, ext = os.path.splitext(fitxer)
                i = 1
                while os.path.exists(os.path.join(carpeta_destinacio, f"{nom}_{i}{ext}")):
                    i += 1
                cami_destinacio = os.path.join(carpeta_destinacio, f"{nom}_{i}{ext}")

            shutil.move(cami_actual, cami_destinacio)

        # Elimina subcarpetes buides
        for subcarpeta in dirs:
            cami_subcarpeta = os.path.join(arrel, subcarpeta)
            if os.path.isdir(cami_subcarpeta) and not os.listdir(cami_subcarpeta):
                os.rmdir(cami_subcarpeta)

if __name__ == "__main__":
    directori = os.getcwd()  # Agafa la ruta actual
    mou_arxius_a_indian_food(directori)
    print(f"Tots els fitxers s'han mogut a la carpeta: '{os.path.join(directori, 'Indian Food Generalitzat')}'")
