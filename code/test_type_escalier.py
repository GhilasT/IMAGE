import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
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
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return None
    
    try:
        # On execute toute les étapes de détéction d'escalier
        resultats = detection_escaliers_complete(image_path, visualiser=visualiser)
        
        # Afficher le résultat
        if resultats and "type_escalier" in resultats:
            type_escalier = resultats["type_escalier"]
            print(f"\n=== RÉSULTAT: ESCALIER {type_escalier.upper()} ===\n")
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

def tester_images_labelisees(dossier_images, visualiser=False):
    """
    Teste la détection du type d'escalier sur un ensemble d'images labelisées
    et retourne des statistiques de performance.
    
    Args:
        dossier_images (str): Chemin vers le dossier contenant les images et leurs JSON
        visualiser (bool): Si True, affiche des visualisations
    
    Returns:
        dict: Statistiques de performance
    """
    from evaluer_performances import evaluer_performances
    resultats = evaluer_performances(dossier_images, visualiser)
    
    return resultats

def analyser_resultats_detailles(resultats_df_path):
    """
    Analyse un fichier CSV de résultats de détection pour voir quels motifs
    mènent à des classifications incorrectes.
    
    Args:
        resultats_df_path (str): Chemin vers le fichier CSV de résultats
        
    Returns:
        None: Affiche directement des statistiques et graphiques
    """
    # Charger le CSV
    try:
        df = pd.read_csv(resultats_df_path)
        print(f"Fichier chargé avec {len(df)} entrées")
        
        # On calcule les statistiques
        total = len(df)
        corrects = df['correct'].sum()
        incorrects = total - corrects
        
        print(f"\n=== Analyse des résultats ===")
        print(f"Prédictions correctes: {corrects} ({corrects/total*100:.1f}%)")
        print(f"Prédictions incorrectes: {incorrects} ({incorrects/total*100:.1f}%)")
        
        # Analyse par type réel
        types_reels = df['label_reel'].value_counts()
        print("\n=== Distribution des types réels ===")
        for type_escalier, count in types_reels.items():
            print(f"- {type_escalier}: {count} ({count/total*100:.1f}%)")
        
        # Analyse des erreurs
        erreurs = df[df['correct'] == False]
        if not erreurs.empty:
            print("\n=== Analyse des erreurs ===")
            
            # Erreurs par type
            erreurs_par_type = erreurs.groupby(['label_reel', 'prediction']).size()
            print(erreurs_par_type)
            
            # Taux d'erreur par type réel
            for type_escalier in types_reels.index:
                nb_ce_type = len(df[df['label_reel'] == type_escalier])
                nb_erreurs = len(erreurs[erreurs['label_reel'] == type_escalier])
                taux_erreur = nb_erreurs / nb_ce_type if nb_ce_type > 0 else 0
                print(f"Taux d'erreur pour les escaliers {type_escalier}: {taux_erreur*100:.1f}%")
                
            # Liste des fichiers mal classifiés
            print("\nFichiers mal classifiés:")
            for _, row in erreurs.iterrows():
                print(f"- {row['fichier']}: Réel {row['label_reel']}, Prédit {row['prediction']}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse du fichier: {e}")

if __name__ == "__main__":
    # Chemin vers une image d'escalier à tester
    image_path = "images/14.jpg" 
    
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
    
    # Optionnel: Test des images labelisées
    # dossier_labelise = "../Labelised_stairs_direction/"
    # tester_images_labelisees(dossier_labelise, visualiser=True)
    
    # Optionnel: Analyser un rapport existant
    # analyser_resultats_detailles("rapport_performance.csv")
