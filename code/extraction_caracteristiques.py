import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from extraction_lignes import extraction_lignes_simples
from pretraitement import pretraitement_image
from detection_contours import detection_contours

def extraction_caracteristiques(lignes_horizontales, lignes_verticales, dimensions_image, visualiser=False):
    """
    Calcule les caractéristiques importantes des lignes détectées pour la classification d'escaliers.
    
    Args:
        lignes_horizontales: Liste des lignes horizontales détectées
        lignes_verticales: Liste des lignes verticales détectées
        dimensions_image: Tuple (hauteur, largeur) de l'image
        visualiser: Si True, affiche des visualisations des caractéristiques
        
    Returns:
        dict: Dictionnaire contenant les caractéristiques extraites
    """
    hauteur, largeur = dimensions_image[:2]
    caracteristiques = {}
    
    # 1. Calculer le rapport entre lignes horizontales et verticales
    nb_horizontales = len(lignes_horizontales)
    nb_verticales = len(lignes_verticales)
    
    if nb_verticales > 0:
        ratio_h_v = nb_horizontales / nb_verticales
    else:
        ratio_h_v = float('inf') if nb_horizontales > 0 else 0
    
    caracteristiques["ratio_h_v"] = ratio_h_v
    caracteristiques["nb_horizontales"] = nb_horizontales
    caracteristiques["nb_verticales"] = nb_verticales
    
    # 2. Mesurer l'espacement entre les lignes (périodicité)
    # Extraire les coordonnées y des lignes horizontales
    y_coords = []
    for ligne in lignes_horizontales:
        x1, y1, x2, y2 = ligne[0]
        y_moyen = (y1 + y2) / 2
        y_coords.append(y_moyen)
    
    # Trier les coordonnées y
    y_coords.sort()
    
    # Calculer les espacements entre lignes horizontales consécutives
    espacements = []
    for i in range(1, len(y_coords)):
        espacements.append(y_coords[i] - y_coords[i-1])
    
    if espacements:
        caracteristiques["espacement_min"] = min(espacements)
        caracteristiques["espacement_max"] = max(espacements)
        caracteristiques["espacement_moyen"] = np.mean(espacements)
        caracteristiques["espacement_ecart_type"] = np.std(espacements)
        
        # Calculer la périodicité (régularité de l'espacement)
        if len(espacements) >= 2:
            variation_espacement = caracteristiques["espacement_ecart_type"] / caracteristiques["espacement_moyen"]
            caracteristiques["periodicite"] = 1 - min(1, variation_espacement)  # Plus proche de 1 = plus périodique
        else:
            caracteristiques["periodicite"] = 0
    else:
        caracteristiques["espacement_min"] = 0
        caracteristiques["espacement_max"] = 0
        caracteristiques["espacement_moyen"] = 0
        caracteristiques["espacement_ecart_type"] = 0
        caracteristiques["periodicite"] = 0
    
    # 3. Analyser la distribution des longueurs de lignes
    longueurs_h = []
    for ligne in lignes_horizontales:
        x1, y1, x2, y2 = ligne[0]
        longueur = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        longueurs_h.append(longueur)
    
    longueurs_v = []
    for ligne in lignes_verticales:
        x1, y1, x2, y2 = ligne[0]
        longueur = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        longueurs_v.append(longueur)
    
    # Caractéristiques de longueur des lignes horizontales
    if longueurs_h:
        caracteristiques["longueur_h_min"] = min(longueurs_h)
        caracteristiques["longueur_h_max"] = max(longueurs_h)
        caracteristiques["longueur_h_moyenne"] = np.mean(longueurs_h)
        caracteristiques["longueur_h_ecart_type"] = np.std(longueurs_h)
        caracteristiques["longueur_h_mediane"] = np.median(longueurs_h)
    else:
        caracteristiques["longueur_h_min"] = 0
        caracteristiques["longueur_h_max"] = 0
        caracteristiques["longueur_h_moyenne"] = 0
        caracteristiques["longueur_h_ecart_type"] = 0
        caracteristiques["longueur_h_mediane"] = 0
    
    # Caractéristiques de longueur des lignes verticales
    if longueurs_v:
        caracteristiques["longueur_v_min"] = min(longueurs_v)
        caracteristiques["longueur_v_max"] = max(longueurs_v)
        caracteristiques["longueur_v_moyenne"] = np.mean(longueurs_v)
        caracteristiques["longueur_v_ecart_type"] = np.std(longueurs_v)
        caracteristiques["longueur_v_mediane"] = np.median(longueurs_v)
    else:
        caracteristiques["longueur_v_min"] = 0
        caracteristiques["longueur_v_max"] = 0
        caracteristiques["longueur_v_moyenne"] = 0
        caracteristiques["longueur_v_ecart_type"] = 0
        caracteristiques["longueur_v_mediane"] = 0
    
    # 4. Analyse de la distribution spatiale des lignes
    if y_coords:
        caracteristiques["y_min"] = min(y_coords)
        caracteristiques["y_max"] = max(y_coords)
        caracteristiques["couverture_verticale"] = (max(y_coords) - min(y_coords)) / hauteur if hauteur > 0 else 0
    else:
        caracteristiques["y_min"] = 0
        caracteristiques["y_max"] = 0
        caracteristiques["couverture_verticale"] = 0
    
    # Visualisation des caractéristiques
    if visualiser and y_coords:
        plt.figure(figsize=(15, 10))
        
        # Histogramme des espacements
        plt.subplot(2, 2, 1)
        plt.title("Distribution des espacements entre lignes horizontales")
        plt.hist(espacements, bins=10, color='blue', alpha=0.7)
        plt.xlabel("Espacement (pixels)")
        plt.ylabel("Fréquence")
        plt.axvline(caracteristiques["espacement_moyen"], color='red', linestyle='dashed', 
                   linewidth=2, label=f"Moyen: {caracteristiques['espacement_moyen']:.1f}")
        plt.legend()
        
        # Histogramme des longueurs horizontales
        plt.subplot(2, 2, 2)
        plt.title("Distribution des longueurs des lignes horizontales")
        plt.hist(longueurs_h, bins=10, color='green', alpha=0.7)
        plt.xlabel("Longueur (pixels)")
        plt.ylabel("Fréquence")
        plt.axvline(caracteristiques["longueur_h_moyenne"], color='red', linestyle='dashed', 
                   linewidth=2, label=f"Moyen: {caracteristiques['longueur_h_moyenne']:.1f}")
        plt.legend()
        
        # Histogramme des longueurs verticales
        plt.subplot(2, 2, 3)
        plt.title("Distribution des longueurs des lignes verticales")
        if longueurs_v:
            plt.hist(longueurs_v, bins=10, color='red', alpha=0.7)
            plt.axvline(caracteristiques["longueur_v_moyenne"], color='blue', linestyle='dashed', 
                       linewidth=2, label=f"Moyen: {caracteristiques['longueur_v_moyenne']:.1f}")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Pas de lignes verticales", 
                    horizontalalignment='center', verticalalignment='center')
        plt.xlabel("Longueur (pixels)")
        plt.ylabel("Fréquence")
        
        # Visualisation de la position des lignes
        plt.subplot(2, 2, 4)
        plt.title("Position des lignes horizontales")
        for i, y in enumerate(y_coords):
            plt.axhline(y, color='blue', alpha=0.5)
            plt.text(10, y+5, f"{y:.0f}", color='blue')
        
        plt.ylim(0, hauteur)
        plt.xlabel("Position x")
        plt.ylabel("Position y")
        plt.gca().invert_yaxis()  # Pour correspondre à la convention d'image (0,0 en haut à gauche)
        
        plt.tight_layout()
        plt.show()
        
        # Tableau récapitulatif des caractéristiques
        print("\n=== Caractéristiques extraites ===")
        print(f"Rapport lignes H/V: {ratio_h_v:.2f}")
        print(f"Nombre de lignes horizontales: {nb_horizontales}")
        print(f"Nombre de lignes verticales: {nb_verticales}")
        print(f"Espacement moyen entre lignes horizontales: {caracteristiques['espacement_moyen']:.2f} pixels")
        print(f"Périodicité des espacements: {caracteristiques['periodicite']:.2f} (1 = parfait)")
        print(f"Longueur moyenne des lignes horizontales: {caracteristiques['longueur_h_moyenne']:.2f} pixels")
        print(f"Couverture verticale de l'image: {caracteristiques['couverture_verticale']*100:.1f}%")
    
    return caracteristiques

def classification_escaliers(image, caracteristiques, visualiser=False):
    """
    Classifie les zones d'image comme escaliers ou non en utilisant les caractéristiques extraites
    et des règles heuristiques
    
    Args:
        image: Image originale
        caracteristiques: Dictionnaire des caractéristiques extraites
        visualiser: Si True, affiche les résultats de la classification
        
    Returns:
        dict: Résultats de la classification avec scores et image segmentée
    """
    hauteur, largeur = image.shape[:2]
    
    # Créer une image pour la visualisation de la classification
    image_classification = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Application d'un seuillage Otsu pour segmenter l'image
    if len(image.shape) == 3:
        image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gris = image.copy()
    
    # Appliquer le seuillage d'Otsu
    _, image_seuil = cv2.threshold(image_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Créer un masque pour les zones d'escaliers potentiels
    masque_escalier = np.zeros((hauteur, largeur), dtype=np.uint8)
    
    # 1. Règles heuristiques basées sur les caractéristiques extraites
    score_escalier = 0
    est_escalier = False
    raisons = []
    
    # Règle 1: Nombre minimum de lignes horizontales
    if caracteristiques["nb_horizontales"] >= 3:
        score_escalier += 30
        raisons.append(f"Nombre suffisant de lignes horizontales ({caracteristiques['nb_horizontales']})")
    else:
        raisons.append(f"Nombre insuffisant de lignes horizontales ({caracteristiques['nb_horizontales']} < 3)")
    
    # Règle 2: Rapport entre lignes horizontales et verticales
    ratio_ideal = 1.5  # Un escalier a généralement plus de lignes horizontales que verticales
    if 0.8 <= caracteristiques["ratio_h_v"] <= 3.0:
        score_escalier += 15
        raisons.append(f"Bon rapport H/V ({caracteristiques['ratio_h_v']:.2f})")
    else:
        raisons.append(f"Rapport H/V non idéal ({caracteristiques['ratio_h_v']:.2f})")
    
    # Règle 3: Périodicité des espacements entre lignes horizontales
    if caracteristiques["periodicite"] > 0.7:
        score_escalier += 25
        raisons.append(f"Bonne périodicité des espacements ({caracteristiques['periodicite']:.2f})")
    elif caracteristiques["periodicite"] > 0.5:
        score_escalier += 15
        raisons.append(f"Périodicité moyenne des espacements ({caracteristiques['periodicite']:.2f})")
    else:
        raisons.append(f"Faible périodicité des espacements ({caracteristiques['periodicite']:.2f})")
    
    # Règle 4: Couverture verticale de l'image
    if caracteristiques["couverture_verticale"] > 0.4:
        score_escalier += 15
        raisons.append(f"Bonne couverture verticale ({caracteristiques['couverture_verticale']*100:.1f}%)")
    else:
        raisons.append(f"Faible couverture verticale ({caracteristiques['couverture_verticale']*100:.1f}%)")
    
    # Règle 5: Uniformité des longueurs de lignes horizontales
    if caracteristiques["longueur_h_ecart_type"] > 0 and caracteristiques["longueur_h_moyenne"] > 0:
        coeff_variation = caracteristiques["longueur_h_ecart_type"] / caracteristiques["longueur_h_moyenne"]
        if coeff_variation < 0.3:
            score_escalier += 15
            raisons.append(f"Bonne uniformité des longueurs de marches (CV={coeff_variation:.2f})")
        elif coeff_variation < 0.5:
            score_escalier += 10
            raisons.append(f"Uniformité moyenne des longueurs de marches (CV={coeff_variation:.2f})")
        else:
            raisons.append(f"Faible uniformité des longueurs de marches (CV={coeff_variation:.2f})")
    
    # Décision finale
    if score_escalier >= 70:
        est_escalier = True
        confiance = "Élevée"
    elif score_escalier >= 50:
        est_escalier = True
        confiance = "Moyenne"
    elif score_escalier >= 30:
        est_escalier = False
        confiance = "Faible"
    else:
        est_escalier = False
        confiance = "Très faible"
    
    # 2. Si c'est un escalier, essayer de délimiter la zone d'escalier
    if est_escalier:
        # Créer un masque basé sur les lignes horizontales et verticales
        y_coords = []
        for ligne in caracteristiques.get("lignes_horizontales", []):
            x1, y1, x2, y2 = ligne[0]
            y_moyen = int((y1 + y2) / 2)
            y_coords.append(y_moyen)
        
        if y_coords:
            y_min = max(0, int(min(y_coords) - caracteristiques["espacement_moyen"]))
            y_max = min(hauteur, int(max(y_coords) + caracteristiques["espacement_moyen"]))
            
            # Trouver les limites horizontales
            x_coords = []
            for ligne in caracteristiques.get("lignes_horizontales", []):
                x1, y1, x2, y2 = ligne[0]
                x_coords.extend([x1, x2])
            
            if x_coords:
                x_min = max(0, int(min(x_coords) - 20))
                x_max = min(largeur, int(max(x_coords) + 20))
                
                # Créer le masque et le visualiser
                masque_escalier[y_min:y_max, x_min:x_max] = 255
                
                # Dessiner la zone d'escalier sur l'image de classification
                cv2.rectangle(image_classification, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image_classification, f"ESCALIER ({score_escalier}%)", (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    # Visualisation des résultats
    if visualiser:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Image originale")
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Seuillage Otsu")
        plt.imshow(image_seuil, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Masque d'escalier")
        plt.imshow(masque_escalier, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title(f"Classification: {'ESCALIER' if est_escalier else 'PAS D\'ESCALIER'}\nScore: {score_escalier}% - Confiance: {confiance}")
        plt.imshow(cv2.cvtColor(image_classification, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les raisons de la classification
        print("\n=== Résultats de la classification ===")
        print(f"Score total: {score_escalier}%")
        print(f"Décision: {'ESCALIER' if est_escalier else 'PAS D\'ESCALIER'}")
        print(f"Niveau de confiance: {confiance}")
        print("\nRaisons:")
        for i, raison in enumerate(raisons, 1):
            print(f"  {i}. {raison}")
    
    resultats = {
        "est_escalier": est_escalier,
        "score": score_escalier,
        "confiance": confiance,
        "raisons": raisons,
        "masque_escalier": masque_escalier,
        "image_classification": image_classification
    }
    
    return resultats

def detection_escaliers_complete(image_path, visualiser=False):
    """
    Pipeline complet pour la détection d'escaliers:
    1. Prétraitement de l'image
    2. Détection des contours
    3. Extraction des lignes droites
    4. Extraction des caractéristiques
    5. Classification
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        visualiser (bool): Si True, affiche les résultats de chaque étape
    
    Returns:
        dict: Résultats de la classification
    """
    # Charger l'image originale
    image_originale = cv2.imread(image_path)
    if image_originale is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    
    print("1. Prétraitement de l'image...")
    image_pretraitee = pretraitement_image(image_path, visualiser=visualiser)
    
    print("2. Détection des contours...")
    contours = detection_contours(image_pretraitee, visualiser=visualiser)
    
    print("3. Extraction des lignes droites...")
    resultats_lignes = extraction_lignes_simples(contours, visualiser=visualiser)
    
    if resultats_lignes:
        print("4. Extraction des caractéristiques...")
        dimensions_image = image_originale.shape
        caracteristiques = extraction_caracteristiques(
            resultats_lignes["lignes_horizontales"],
            resultats_lignes["lignes_verticales"],
            dimensions_image,
            visualiser=visualiser
        )
        
        # Ajouter les listes de lignes aux caractéristiques pour la classification
        caracteristiques["lignes_horizontales"] = resultats_lignes["lignes_horizontales"]
        caracteristiques["lignes_verticales"] = resultats_lignes["lignes_verticales"]
        
        print("5. Classification...")
        resultats = classification_escaliers(image_originale, caracteristiques, visualiser=visualiser)
        
        # Sauvegarder les résultats
        cv2.imwrite("resultats_classification.jpg", resultats["image_classification"])
        cv2.imwrite("masque_escalier.jpg", resultats["masque_escalier"])
        
        print(f"\nRésultat: {'ESCALIER DÉTECTÉ' if resultats['est_escalier'] else 'PAS D\'ESCALIER'}")
        print(f"Score: {resultats['score']}%")
        print(f"Confiance: {resultats['confiance']}")
        
        return resultats
    else:
        print("Aucune ligne détectée, impossible de poursuivre l'analyse.")
        return None

if __name__ == "__main__":
    # Chemin vers une image d'escalier
    image_path = "images/14.jpg"
    
    try:
        # Exécuter le pipeline complet
        resultats = detection_escaliers_complete(image_path, visualiser=True)
    except Exception as e:
        print(f"Erreur lors de la détection: {e}")