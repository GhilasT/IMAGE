import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from extraction_caracteristiques import detection_escaliers_complete

def tester_type_escalier(image_path, visualiser=True):
    """
    Teste la détection du type d'escalier sur une image spécifique
    et affiche des informations détaillées pour le débogage.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        visualiser (bool): Si True, affiche des visualisations détaillées
    
    Returns:
        dict: Résultats de la détection
    """
    print(f"\nTest de détection sur: {image_path}")
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return None
    
    try:
        # Exécuter la détection complète
        resultats = detection_escaliers_complete(image_path, visualiser=visualiser)
        
        # Afficher le résultat
        if resultats and "type_escalier" in resultats:
            type_escalier = resultats["type_escalier"]
            print(f"\n=== RÉSULTAT: ESCALIER {type_escalier.upper()} ===\n")
            
            # Afficher l'image avec l'annotation
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(resultats["image_classification"], cv2.COLOR_BGR2RGB))
            plt.title(f"Escalier détecté: {type_escalier.upper()}")
            plt.axis('off')
            plt.show()
            
            return resultats
        else:
            print("Aucun escalier détecté ou erreur de classification.")
            return None
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Chemin vers une image d'escalier à tester
    image_path = "images/14.jpg"  # Remplacez par votre image d'escalier
    
    # Test sur une image spécifique
    tester_type_escalier(image_path, visualiser=True)
    
    # Optionnel: Test sur plusieurs images (décommentez pour utiliser)
    # dossier_images = "images/"
    # extensions = ["*.jpg", "*.jpeg", "*.png"]
    # fichiers_images = []
    # 
    # for extension in extensions:
    #     fichiers_images.extend(glob.glob(os.path.join(dossier_images, extension)))
    # 
    # for img_path in fichiers_images[:5]:  # Tester les 5 premières images
    #     tester_type_escalier(img_path, visualiser=False)
