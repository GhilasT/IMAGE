import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from extraction_caracteristiques import detection_escaliers_complete

def tester_classification_type(dossier_images="images/", visualiser=True):
    """
    Teste la classification du type d'escalier (droit ou tournant) sur plusieurs images.
    
    Args:
        dossier_images (str): Chemin vers le dossier contenant les images
        visualiser (bool): Si True, affiche les résultats pour chaque image
    """
    # Récupérer tous les fichiers images du dossier
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    fichiers_images = []
    
    for extension in extensions:
        fichiers_images.extend(glob.glob(os.path.join(dossier_images, extension)))
    
    print(f"Nombre d'images trouvées: {len(fichiers_images)}")
    resultats = []
    
    # Traiter chaque image
    for i, image_path in enumerate(fichiers_images):
        try:
            print(f"\nTraitement de l'image {i+1}/{len(fichiers_images)}: {image_path}")
            
            # Exécuter le pipeline complet
            resultat = detection_escaliers_complete(image_path, visualiser=visualiser)
            
            if resultat and resultat["est_escalier"]:
                type_escalier = resultat.get("type_escalier", "indéterminé")
                
                resultats.append({
                    "image": os.path.basename(image_path),
                    "type": type_escalier
                })
                
                print(f"Résultat: Escalier {type_escalier.upper()}")
            else:
                print("Aucun escalier détecté dans cette image.")
        
        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Afficher un récapitulatif
    if resultats:
        print("\n=== Récapitulatif des résultats ===")
        print(f"Total d'escaliers détectés: {len(resultats)}")
        
        # Compter les types d'escaliers
        types_escaliers = {}
        for res in resultats:
            type_esc = res["type"]
            types_escaliers[type_esc] = types_escaliers.get(type_esc, 0) + 1
        
        for type_esc, count in types_escaliers.items():
            print(f"Escaliers {type_esc}: {count} ({count/len(resultats)*100:.1f}%)")
    
    return resultats

if __name__ == "__main__":
    try:
        # Tester sur toutes les images du dossier
        resultats = tester_classification_type(dossier_images="images/", visualiser=True)
        
        # Pour tester sur une seule image, décommentez ces lignes:
        #image_path = "images/4.jpg"  # Remplacez par votre image d'escalier
        #resultat = detection_escaliers_complete(image_path, visualiser=True)
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
