import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from extraction_caracteristiques import detection_escaliers_complete
import pandas as pd

def tester_images_escaliers(dossier_images="images/", nombre_max=None, save_results=True):
    """
    Teste la détection du type d'escalier sur plusieurs images et 
    sauvegarde les résultats avec métriques pour analyse.
    
    Args:
        dossier_images (str): Dossier contenant les images d'escaliers
        nombre_max (int): Nombre maximum d'images à tester (None = toutes)
        save_results (bool): Si True, sauvegarde les résultats dans un fichier CSV
    
    Returns:
        pandas.DataFrame: Tableau des résultats
    """
    # Récupérer les chemins des images
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    fichiers_images = []
    
    for extension in extensions:
        fichiers_images.extend(glob.glob(os.path.join(dossier_images, extension)))
    
    if nombre_max:
        fichiers_images = fichiers_images[:nombre_max]
    
    print(f"Test sur {len(fichiers_images)} images...")
    
    # Créer un tableau pour stocker les résultats
    resultats = []
    
    for i, image_path in enumerate(fichiers_images): #Pour chaque images
        try:
            print(f"\n[{i+1}/{len(fichiers_images)}] Traitement de {os.path.basename(image_path)}")
            
            res = detection_escaliers_complete(image_path, visualiser=False)
            
            if res and "type_escalier" in res: # On récupere les métriques importantes
                metriques = res.get("metriques", {})
                ecart_type_angles = metriques.get("ecart_type_angles", float('nan'))
                variation_centres = metriques.get("variation_centres", float('nan'))
                resultats.append({
                    "image": os.path.basename(image_path),
                    "type_escalier": res["type_escalier"],
                    "ecart_type_angles": ecart_type_angles,
                    "variation_centres": variation_centres,
                    "nb_lignes_h": len(res.get("metriques", {}).get("angles", []))
                })
                
                print(f"Résultat: {res['type_escalier'].upper()}")
            else:
                print("Échec de la détection")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if resultats:
        df_resultats = pd.DataFrame(resultats)
        
        print("\n=== Statistiques des résultats ===")
        print(f"Total d'escaliers: {len(df_resultats)}")
        count_droits = sum(df_resultats["type_escalier"] == "droit")
        count_tournants = sum(df_resultats["type_escalier"] == "tournant")
        print(f"Escaliers droits: {count_droits} ({count_droits/len(df_resultats)*100:.1f}%)")
        print(f"Escaliers tournants: {count_tournants} ({count_tournants/len(df_resultats)*100:.1f}%)")
        
        # Statistiques des métriques
        print("\n=== Moyennes des métriques ===")
        droits = df_resultats[df_resultats["type_escalier"] == "droit"]
        tournants = df_resultats[df_resultats["type_escalier"] == "tournant"]
        
        if not droits.empty:
            print(f"Escaliers DROITS:")
            print(f"- Écart-type angles: {droits['ecart_type_angles'].mean():.2f}°")
            print(f"- Variation centres: {droits['variation_centres'].mean():.2f}")
        
        if not tournants.empty:
            print(f"Escaliers TOURNANTS:")
            print(f"- Écart-type angles: {tournants['ecart_type_angles'].mean():.2f}°")
            print(f"- Variation centres: {tournants['variation_centres'].mean():.2f}")
        
        if save_results:
            csv_path = os.path.join(dossier_images, "resultats_detection.csv")
            df_resultats.to_csv(csv_path, index=False)
            print(f"\nRésultats sauvegardés dans {csv_path}")
        
        return df_resultats
    else:
        print("Aucun résultat obtenu.")
        return None

if __name__ == "__main__":
    image_path = "images/15.jpg" 
    print(f"Test sur l'image {image_path}...")
    resultat = detection_escaliers_complete(image_path, visualiser=True)
    
    # Décommentez pour tester sur plusieurs images
    # df = tester_images_escaliers(dossier_images="images/", nombre_max=10)
