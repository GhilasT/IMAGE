import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add these exports at the top of the file
__all__ = ['affiner_contours', 'verifier_coherence_spatiale', 'classifier_type_escalier', 'post_traitement', 'verifier_type_escalier']

def affiner_contours(masque_escalier, kernel_size=(5, 5), iterations=1, visualiser=False):
    """
    Affine les contours détectés en utilisant des opérations morphologiques.
    
    Args:
        masque_escalier (numpy.ndarray): Masque binaire de l'escalier détecté
        kernel_size (tuple): Taille du noyau pour les opérations morphologiques
        iterations (int): Nombre d'itérations pour les opérations morphologiques
        visualiser (bool): Si True, affiche les résultats intermédiaires
        
    Returns:
        numpy.ndarray: Masque affiné
    """
    masque_affine = masque_escalier.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    masque_fermeture = cv2.morphologyEx(masque_affine, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    masque_ouverture = cv2.morphologyEx(masque_fermeture, cv2.MORPH_OPEN, kernel, iterations=iterations)
    masque_dilate = cv2.dilate(masque_ouverture, kernel, iterations=1)
    
    if visualiser:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.title("Masque original")
        plt.imshow(masque_escalier, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.title("Fermeture")
        plt.imshow(masque_fermeture, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.title("Ouverture")
        plt.imshow(masque_ouverture, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.title("Masque affiné")
        plt.imshow(masque_dilate, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return masque_dilate

def verifier_coherence_spatiale(masque_escalier, lignes_horizontales, image_dims, visualiser=False):
    """
    Vérifie la cohérence spatiale de l'escalier détecté en analysant:
    - La distribution des lignes horizontales
    - La forme et les proportions de la région détectée
    - L'alignement des marches
    
    Args:
        masque_escalier (numpy.ndarray): Masque binaire de l'escalier détecté
        lignes_horizontales (list): Liste des lignes horizontales détectées
        image_dims (tuple): Dimensions de l'image (hauteur, largeur)
        visualiser (bool): Si True, affiche les résultats
        
    Returns:
        tuple: (masque_coherent, score_coherence)
    """
    hauteur, largeur = image_dims[:2]
    masque_coherent = masque_escalier.copy()
    score_coherence = 100
    annotations = []
    
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masque_escalier, connectivity=8)
    
    if nb_labels > 1:
        min_area = hauteur * largeur * 0.01
        masque_coherent = np.zeros_like(masque_escalier)
        composantes_valides = 0
        for i in range(1, nb_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                composantes_valides += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                ratio = h / w if w > 0 else 0
                if ratio > 0.5 and ratio < 4.0:
                    masque_coherent[labels == i] = 255
                    annotations.append(f"Composante {i}: {w}×{h} pixels, ratio={ratio:.1f}")
                else:
                    score_coherence -= 20
                    annotations.append(f"Composante {i} rejetée: {w}×{h} pixels, ratio={ratio:.1f}")
        
        if composantes_valides == 0:
            masque_coherent = masque_escalier
            score_coherence -= 30
            annotations.append("Aucune composante valide trouvée après filtrage")
    
    if lignes_horizontales is not None and len(lignes_horizontales) >= 3:
        y_coords = []
        for ligne in lignes_horizontales:
            x1, y1, x2, y2 = ligne[0]
            y_moyen = (y1 + y2) / 2
            y_coords.append(y_moyen)
        y_coords.sort()
        differences = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        if differences:
            ecart_type = np.std(differences)
            moyenne = np.mean(differences)
            cv = ecart_type / moyenne if moyenne > 0 else float('inf')
            if cv < 0.2:
                score_coherence += 10
                annotations.append(f"Lignes très régulières (CV={cv:.2f})")
            elif cv < 0.5:
                score_coherence += 5
                annotations.append(f"Lignes assez régulières (CV={cv:.2f})")
            else:
                score_coherence -= 15
                annotations.append(f"Lignes irrégulières (CV={cv:.2f})")
    
    props = cv2.moments(masque_coherent)
    if props["m00"] > 0:
        cx = int(props["m10"] / props["m00"])
        cy = int(props["m01"] / props["m00"])
        x_ratio = cx / largeur
        y_ratio = cy / hauteur
        if 0.2 <= x_ratio <= 0.8 and 0.2 <= y_ratio <= 0.8:
            score_coherence += 5
            annotations.append(f"Position centrale correcte (x={x_ratio:.2f}, y={y_ratio:.2f})")
        else:
            score_coherence -= 5
            annotations.append(f"Position excentrée (x={x_ratio:.2f}, y={y_ratio:.2f})")
    
    score_coherence = max(0, min(100, score_coherence))
    
    if visualiser:
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title("Masque original")
        plt.imshow(masque_escalier, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(f"Masque cohérent (score: {score_coherence:.1f}%)")
        plt.imshow(masque_coherent, cmap='gray')
        plt.axis('off')
        for i, annotation in enumerate(annotations):
            plt.figtext(0.1, 0.05 - i*0.03, annotation, fontsize=9)
        plt.tight_layout()
        plt.show()
    
    return masque_coherent, score_coherence

def classifier_type_escalier(masque_escalier, lignes_horizontales, image_dims, visualiser=False):
    """
    Détermine si un escalier est droit ou tournant en analysant:
    - La disposition des lignes horizontales (parallèles vs convergentes)
    - La forme globale du contour de l'escalier
    - L'arrangement des marches
    
    Args:
        masque_escalier (numpy.ndarray): Masque binaire de l'escalier détecté
        lignes_horizontales (list): Liste des lignes horizontales détectées
        image_dims (tuple): Dimensions de l'image (hauteur, largeur)
        visualiser (bool): Si True, affiche les résultats de l'analyse
        
    Returns:
        tuple: (type_escalier, score_confiance)
    """
    hauteur, largeur = image_dims[:2]
    type_escalier = "droit"  # Par défaut
    score_confiance = 50
    justifications = []
    
    if lignes_horizontales is None or len(lignes_horizontales) < 3:
        # Même avec peu de lignes, on considère l'escalier comme tournant par défaut
        return "tournant", 50
    
    lignes_info = []
    for ligne in lignes_horizontales:
        x1, y1, x2, y2 = ligne[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        longueur = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        centre_x = (x1 + x2) / 2
        centre_y = (y1 + y2) / 2
        lignes_info.append({
            'points': (x1, y1, x2, y2),
            'angle': angle,
            'longueur': longueur,
            'centre': (centre_x, centre_y)
        })
    
    lignes_info.sort(key=lambda x: x['centre'][1])
    angles = [info['angle'] for info in lignes_info]
    ecart_type_angles = np.std(angles)
    
    if ecart_type_angles < 5.0:
        score_confiance += 30
        justifications.append(f"Lignes parallèles (écart-type des angles: {ecart_type_angles:.2f}°)")
        type_escalier = "droit"
    elif ecart_type_angles < 15.0:
        score_confiance += 10
        justifications.append(f"Légère convergence des lignes (écart-type des angles: {ecart_type_angles:.2f}°)")
    else:
        score_confiance -= 20
        justifications.append(f"Lignes très convergentes (écart-type des angles: {ecart_type_angles:.2f}°)")
        type_escalier = "tournant"
    
    points = []
    for info in lignes_info:
        x1, y1, x2, y2 = info['points']
        points.append((int(x1), int(y1)))
        points.append((int(x2), int(y2)))
    
    if len(points) >= 4:
        points = np.array(points)
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        rect_area = w * h
        if rect_area > 0:
            ratio = hull_area / rect_area
            if ratio > 0.85:
                score_confiance += 20
                justifications.append(f"Forme rectangulaire (ratio: {ratio:.2f})")
                type_escalier = "droit"
            elif ratio < 0.6:
                score_confiance -= 10
                justifications.append(f"Forme irrégulière (ratio: {ratio:.2f})")
                type_escalier = "tournant"
    
    centres_x = [info['centre'][0] for info in lignes_info]
    centres_y = [info['centre'][1] for info in lignes_info]
    
    if len(centres_x) > 2:
        correlation = np.corrcoef(centres_x, centres_y)[0, 1]
        if abs(correlation) > 0.7:
            score_confiance -= 25
            justifications.append(f"Disposition suivant une courbe (corrélation: {correlation:.2f})")
            type_escalier = "tournant"
        else:
            score_confiance += 15
            justifications.append(f"Disposition linéaire (corrélation: {correlation:.2f})")
    
    if len(lignes_info) >= 3:
        vecteurs = []
        for info in lignes_info:
            x1, y1, x2, y2 = info['points']
            vx = x2 - x1
            vy = y2 - y1
            norme = np.sqrt(vx**2 + vy**2)
            if norme > 0:
                vx /= norme
                vy /= norme
                vecteurs.append((vx, vy, x1, y1))
        
        intersections = []
        for i in range(len(vecteurs)):
            for j in range(i+1, len(vecteurs)):
                vx1, vy1, x1, y1 = vecteurs[i]
                vx2, vy2, x2, y2 = vecteurs[j]
                det = vx1*(-vy2) - vy1*(-vx2)
                if abs(det) > 0.01:
                    t1 = ((x2-x1)*(-vy2) - (y2-y1)*(-vx2)) / det
                    ix = x1 + vx1*t1
                    iy = y1 + vy1*t1
                    if -largeur*2 < ix < largeur*2 and -hauteur*2 < iy < hauteur*2:
                        intersections.append((ix, iy))
        
        if intersections:
            intersections = np.array(intersections)
            if len(intersections) > 2:
                mean_x = np.mean(intersections[:, 0])
                mean_y = np.mean(intersections[:, 1])
                std_x = np.std(intersections[:, 0])
                std_y = np.std(intersections[:, 1])
                if std_x < largeur/4 and std_y < hauteur/4:
                    score_confiance -= 30
                    justifications.append(f"Point de convergence détecté (σx={std_x:.1f}, σy={std_y:.1f})")
                    type_escalier = "tournant"
                    if 0 <= mean_x < largeur and 0 <= mean_y < hauteur:
                        score_confiance -= 10
                        justifications.append("Point de convergence dans l'image")
    
    score_confiance = max(0, min(100, score_confiance))
    
    # Décision finale modifiée pour ne jamais retourner "indéterminé"
    if score_confiance > 50:
        type_escalier = "droit"
    else:
        type_escalier = "tournant"
    
    if visualiser:
        image_visualisation = cv2.cvtColor(masque_escalier.copy(), cv2.COLOR_GRAY2BGR)
        for info in lignes_info:
            x1, y1, x2, y2 = info['points']
            cv2.line(image_visualisation, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        titre = f"Escalier {type_escalier.upper()}"
        cv2.putText(image_visualisation, titre, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image_visualisation, f"Score: {score_confiance:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image_visualisation, cv2.COLOR_BGR2RGB))
        plt.title(f"Classification: Escalier {type_escalier} (confiance: {score_confiance:.1f}%)")
        plt.axis('off')
        for i, justification in enumerate(justifications):
            plt.figtext(0.1, 0.1 - i*0.03, justification, fontsize=9)
        plt.tight_layout()
        plt.show()
    
    return type_escalier, score_confiance

def verifier_type_escalier(resultats, image_originale, caracteristiques, visualiser=False, type_initial=None):
    """
    Vérifie le type d'escalier (droit ou tournant) en supposant toujours que l'escalier existe.
    
    Args:
        resultats (dict): Résultats de la classification d'escalier
        image_originale (numpy.ndarray): Image originale
        caracteristiques (dict): Caractéristiques extraites
        visualiser (bool): Si True, affiche les résultats
        type_initial (tuple): Tuple (type, score) avant post-traitement si disponible
        
    Returns:
        dict: Résultats de la vérification du type d'escalier
    """
    masque_affine = affiner_contours(
        resultats["masque_escalier"],
        kernel_size=(5, 5),
        iterations=2,
        visualiser=False
    )
    
    masque_coherent, _ = verifier_coherence_spatiale(
        masque_affine,
        caracteristiques.get("lignes_horizontales"),
        image_originale.shape,
        visualiser=False
    )
    
    type_escalier, score_type = classifier_type_escalier(
        masque_coherent,
        caracteristiques.get("lignes_horizontales"),
        image_originale.shape,
        visualiser=visualiser
    )
    
    image_resultat = image_originale.copy()
    contours_masque, _ = cv2.findContours(masque_coherent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_resultat, contours_masque, -1, (0, 255, 0), 2)
    
    if contours_masque:
        c = max(contours_masque, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(
            image_resultat, 
            f"ESCALIER {type_escalier.upper()} ({score_type:.1f}%)", 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
    
    if visualiser:
        print("\n=== Résultats de la classification du type d'escalier ===")
        
        # Afficher le type d'escalier avant post-traitement si disponible
        if type_initial:
            type_avant, score_avant = type_initial
            print(f"Type d'escalier AVANT post-traitement: {type_avant.upper()} (score: {score_avant:.1f}%)")
        
        print(f"Type d'escalier APRÈS post-traitement: {type_escalier.upper()} (score: {score_type:.1f}%)")
        
        # Afficher une comparaison si le type a changé
        if type_initial and type_avant != type_escalier:
            print(f"NOTE: Le type d'escalier a changé après le post-traitement!")
    
    resultats_type = resultats.copy()
    resultats_type["est_escalier"] = True  # On suppose toujours que l'escalier existe
    resultats_type["type_escalier"] = type_escalier
    resultats_type["score_type"] = score_type
    resultats_type["image_classification"] = image_resultat
    
    return resultats_type

def post_traitement(resultats, image_originale, caracteristiques, visualiser=False):
    """
    Applique un post-traitement uniquement pour déterminer le type d'escalier.
    Supprime la vérification d'existence d'escalier et suppose toujours que l'escalier existe.
    
    Args:
        resultats (dict): Résultats de la classification d'escalier
        image_originale (numpy.ndarray): Image originale
        caracteristiques (dict): Caractéristiques extraites
        visualiser (bool): Si True, affiche les résultats
        
    Returns:
        dict: Résultats post-traités avec le type d'escalier uniquement
    """
    # Récupérer le type d'escalier avant post-traitement
    type_initial = None
    if "type_escalier" in resultats:
        type_initial = (resultats["type_escalier"], resultats.get("score_type", 0))
    
    if visualiser:
        print("\n" + "="*50)
        print("MENU DE POST-TRAITEMENT")
        print("="*50)
        print("Classification du type d'escalier (droit ou tournant)")
        print("="*50)
    
    # On suppose toujours que l'escalier existe
    resultats["est_escalier"] = True
    
    # Vérifier uniquement le type d'escalier
    resultats_type = verifier_type_escalier(
        resultats, 
        image_originale, 
        caracteristiques, 
        visualiser=visualiser, 
        type_initial=type_initial
    )
    
    return resultats_type

if __name__ == "__main__":
    try:
        from extraction_caracteristiques import detection_escaliers_complete
        
        image_path = "images/.jpg"
        
        print("Détection d'escaliers...")
        resultats = detection_escaliers_complete(image_path, visualiser=False)
        
        if resultats:
            image = cv2.imread(image_path)
            
            print("Application du post-traitement...")
            resultats_post_traites = post_traitement(resultats, image, {}, visualiser=True)
            
            cv2.imwrite("resultat_post_traitement.jpg", resultats_post_traites["image_classification"])
            
            if "masque_escalier" in resultats_post_traites:
                cv2.imwrite("masque_post_traitement.jpg", resultats_post_traites["masque_escalier"])
            
            print("Résultats sauvegardés!")
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
