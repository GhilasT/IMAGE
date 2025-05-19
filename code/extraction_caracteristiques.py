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
    
    # On calcule le rapport entre les  lignes horizontales et verticales
    nb_horizontales = len(lignes_horizontales)
    nb_verticales = len(lignes_verticales)
    
    if nb_verticales > 0:
        ratio_h_v = nb_horizontales / nb_verticales
    else:
        ratio_h_v = float('inf') if nb_horizontales > 0 else 0
    
    caracteristiques["ratio_h_v"] = ratio_h_v
    caracteristiques["nb_horizontales"] = nb_horizontales
    caracteristiques["nb_verticales"] = nb_verticales
    
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
        
        if len(espacements) >= 2:
            variation_espacement = caracteristiques["espacement_ecart_type"] / caracteristiques["espacement_moyen"]
            caracteristiques["periodicite"] = 1 - min(1, variation_espacement)
        else:
            caracteristiques["periodicite"] = 0
    else:
        caracteristiques["espacement_min"] = 0
        caracteristiques["espacement_max"] = 0
        caracteristiques["espacement_moyen"] = 0
        caracteristiques["espacement_ecart_type"] = 0
        caracteristiques["periodicite"] = 0
    
    # On analyse comment est distribué les longueurs des lignes
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
    
    # Caractéristiques des lignes horizontales
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
    
    # Caractéristiques des lignes verticales
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
    Prépare simplement les données pour la classification du type d'escalier (droit ou tournant)
    en supposant toujours que l'image contient un escalier.
    
    Args:
        image: Image originale
        caracteristiques: Dictionnaire des caractéristiques extraites
        visualiser: Si True, affiche les résultats de la préparation
        
    Returns:
        dict: Données préparées pour la classification du type d'escalier
    """
    hauteur, largeur = image.shape[:2]
    
    # Créer une image pour la visualisation
    image_classification = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # On utilise Otsu pour segmenter l'image
    if len(image.shape) == 3:
        image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gris = image.copy()
    
    _, image_seuil = cv2.threshold(image_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    masque_escalier = np.zeros((hauteur, largeur), dtype=np.uint8)
    
    y_coords = []
    for ligne in caracteristiques.get("lignes_horizontales", []):
        if ligne is not None:
            x1, y1, x2, y2 = ligne[0]
            y_moyen = int((y1 + y2) / 2)
            y_coords.append(y_moyen)
    
    if y_coords:
        # Estimer l'espacement moyen s'il n'est pas disponible
        espacement_moyen = caracteristiques.get("espacement_moyen", 20)
        
        y_min = max(0, int(min(y_coords) - espacement_moyen))
        y_max = min(hauteur, int(max(y_coords) + espacement_moyen))
        
        # Trouver les limites horizontales
        x_coords = []
        for ligne in caracteristiques.get("lignes_horizontales", []):
            if ligne is not None:
                x1, y1, x2, y2 = ligne[0]
                x_coords.extend([x1, x2])
        
        if x_coords:
            x_min = max(0, int(min(x_coords) - 20))
            x_max = min(largeur, int(max(x_coords) + 20))
            
            masque_escalier[y_min:y_max, x_min:x_max] = 255
            
            # Dessiner la zone d'escalier sur l'image de classification
            cv2.rectangle(image_classification, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_classification, "ESCALIER", (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # S'il n'y a pas de lignes, on utilise toute l'image
        masque_escalier = np.ones((hauteur, largeur), dtype=np.uint8) * 255
    
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
        plt.title("Zone d'escalier détectée")
        plt.imshow(cv2.cvtColor(image_classification, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    resultats = {
        "est_escalier": True,
        "masque_escalier": masque_escalier,
        "image_classification": image_classification
    }
    
    return resultats

def detection_escaliers_complete(image_path, visualiser=False):
    """
    Pipeline complet pour la détection du type d'escalier:
    1. Prétraitement de l'image
    2. Détection des contours
    3. Extraction des lignes droites
    4. Extraction des caractéristiques
    5. Préparation des données (suppose toujours que l'escalier existe)
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        visualiser (bool): Si True, affiche les résultats de chaque étape
    
    Returns:
        dict: Résultats de la classification du type d'escalier
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
    
    if resultats_lignes is None:
        print("Aucune ligne détectée. On suppose un escalier tournant par défaut.")
        # Créer un résultat par défaut avec escalier tournant
        resultats = {
            "est_escalier": True,
            "type_escalier": "tournant",
            "masque_escalier": np.zeros((image_originale.shape[0], image_originale.shape[1]), dtype=np.uint8),
            "image_classification": image_originale.copy()
        }
        return resultats
    
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
    
    print("5. Préparation pour la classification du type...")
    resultats = classification_escaliers(image_originale, caracteristiques, visualiser=visualiser)
    
    # Déterminer directement le type d'escalier ici au lieu d'utiliser le post-traitement
    if "lignes_horizontales" in caracteristiques and len(caracteristiques["lignes_horizontales"]) >= 3:
        lignes_h = caracteristiques["lignes_horizontales"]
        
        lignes_avec_longueur = []
        for ligne in lignes_h:
            x1, y1, x2, y2 = ligne[0]
            longueur = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            lignes_avec_longueur.append((ligne, longueur))
        
        lignes_avec_longueur.sort(key=lambda x: x[1], reverse=True)
        
        n_lignes_principales = min(len(lignes_avec_longueur), 10)
        lignes_principales = lignes_avec_longueur[:n_lignes_principales]
        if visualiser:
            img_lignes_principales = image_originale.copy()
            for ligne, _ in lignes_principales:
                x1, y1, x2, y2 = ligne[0]
                cv2.line(img_lignes_principales, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            plt.figure(figsize=(10, 8))
            plt.title(f"Les {n_lignes_principales} lignes principales de l'escalier")
            plt.imshow(cv2.cvtColor(img_lignes_principales, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        # On analyse la géométrie des lignes principales
        angles = []
        longueurs = []
        centres_x = []
        
        for ligne, longueur in lignes_principales:
            x1, y1, x2, y2 = ligne[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            centre_x = (x1 + x2) / 2
            
            angles.append(angle)
            longueurs.append(longueur)
            centres_x.append(centre_x)
        
        # On calcule les métriques de classification
        ecart_type_angles = np.std(angles)
        ecart_type_centres = np.std(centres_x)
        longueur_moyenne = np.mean(longueurs)
        variation_centres = ecart_type_centres / longueur_moyenne if longueur_moyenne > 0 else float('inf')
        
        print(f"Métriques des lignes principales:")
        print(f"- Nombre de lignes principales: {n_lignes_principales}")
        print(f"- Écart-type des angles: {ecart_type_angles:.2f}°")
        print(f"- Variation des centres X: {variation_centres:.2f}")
        print(f"- Angles des lignes principales: {[round(a, 1) for a in angles]}")
        
        if ecart_type_angles < 10.0 and variation_centres < 0.3:
            resultats["type_escalier"] = "droit"
            print("Critères d'escalier droit satisfaits!")
        else:
            resultats["type_escalier"] = "tournant"
            print("Critères d'escalier tournant satisfaits!")
            
        resultats["metriques"] = {
            "ecart_type_angles": ecart_type_angles,
            "variation_centres": variation_centres,
            "angles": angles,
            "longueurs": longueurs
        }
    else:
        resultats["type_escalier"] = "tournant"
    
    # On dessine le type d'escalier sur l'image de classification
    image_resultat = resultats["image_classification"].copy()
    masque = resultats["masque_escalier"]
    
    contours_masque, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_resultat, contours_masque, -1, (0, 255, 0), 2)
    
    if contours_masque:
        c = max(contours_masque, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(
            image_resultat, 
            f"ESCALIER {resultats['type_escalier'].upper()}", 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
    
    resultats["image_classification"] = image_resultat
    
    cv2.imwrite("resultats_classification_finale.jpg", resultats["image_classification"])
    cv2.imwrite("masque_escalier_final.jpg", resultats["masque_escalier"])
    
    print(f"\nType d'escalier: {resultats['type_escalier'].upper()}")
    
    return resultats

if __name__ == "__main__":
    # Chemin vers une image d'escalier
    image_path = "images/14.jpg"
    
    try:
        # Exécuter tout le processus
        resultats = detection_escaliers_complete(image_path, visualiser=True)
    except Exception as e:
        print(f"Erreur lors de la détection: {e}")