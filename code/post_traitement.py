import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    # Créer une copie du masque pour éviter de modifier l'original
    masque_affine = masque_escalier.copy()
    
    # Créer le noyau pour les opérations morphologiques
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Fermeture morphologique pour combler les petits trous
    masque_fermeture = cv2.morphologyEx(masque_affine, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Ouverture morphologique pour éliminer les petits objets
    masque_ouverture = cv2.morphologyEx(masque_fermeture, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Dilatation pour agrandir légèrement la région détectée
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
    score_coherence = 100  # Score initial parfait
    annotations = []
    
    # 1. Vérifier les composantes connexes et éliminer les petites régions
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masque_escalier, connectivity=8)
    
    if nb_labels > 1:  # nb_labels inclut toujours l'arrière-plan (label 0)
        # Filtrer les petites composantes
        min_area = hauteur * largeur * 0.01  # Superficie minimale (1% de l'image)
        masque_coherent = np.zeros_like(masque_escalier)
        
        composantes_valides = 0
        for i in range(1, nb_labels):  # Commencer à 1 pour ignorer l'arrière-plan
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                composantes_valides += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Vérifier les proportions
                ratio = h / w if w > 0 else 0
                if ratio > 0.5 and ratio < 4.0:  # Un escalier a généralement une proportion raisonnable
                    masque_coherent[labels == i] = 255
                    annotations.append(f"Composante {i}: {w}×{h} pixels, ratio={ratio:.1f}")
                else:
                    score_coherence -= 20
                    annotations.append(f"Composante {i} rejetée: {w}×{h} pixels, ratio={ratio:.1f}")
        
        if composantes_valides == 0:
            masque_coherent = masque_escalier  # Restaurer si tout a été filtré
            score_coherence -= 30
            annotations.append("Aucune composante valide trouvée après filtrage")
    
    # 2. Vérifier l'alignement des lignes horizontales
    if lignes_horizontales is not None and len(lignes_horizontales) >= 3:
        # Extraire les coordonnées y moyennes des lignes
        y_coords = []
        for ligne in lignes_horizontales:
            x1, y1, x2, y2 = ligne[0]
            y_moyen = (y1 + y2) / 2
            y_coords.append(y_moyen)
        
        # Trier les coordonnées
        y_coords.sort()
        
        # Calculer les différences entre lignes consécutives
        differences = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        if differences:
            # Calculer l'écart-type des différences pour mesurer la régularité
            ecart_type = np.std(differences)
            moyenne = np.mean(differences)
            cv = ecart_type / moyenne if moyenne > 0 else float('inf')
            
            if cv < 0.2:  # Très régulier
                score_coherence += 10
                annotations.append(f"Lignes très régulières (CV={cv:.2f})")
            elif cv < 0.5:  # Assez régulier
                score_coherence += 5
                annotations.append(f"Lignes assez régulières (CV={cv:.2f})")
            else:  # Irrégulier
                score_coherence -= 15
                annotations.append(f"Lignes irrégulières (CV={cv:.2f})")
    
    # 3. Vérifier la position relative dans l'image
    # Les escaliers sont souvent centrés ou alignés avec les bords
    props = cv2.moments(masque_coherent)
    if props["m00"] > 0:
        cx = int(props["m10"] / props["m00"])
        cy = int(props["m01"] / props["m00"])
        
        # Vérifier si le centre est dans les tiers centraux
        x_ratio = cx / largeur
        y_ratio = cy / hauteur
        
        if 0.2 <= x_ratio <= 0.8 and 0.2 <= y_ratio <= 0.8:
            score_coherence += 5
            annotations.append(f"Position centrale correcte (x={x_ratio:.2f}, y={y_ratio:.2f})")
        else:
            score_coherence -= 5
            annotations.append(f"Position excentrée (x={x_ratio:.2f}, y={y_ratio:.2f})")
    
    # Normaliser le score final
    score_coherence = max(0, min(100, score_coherence))
    
    # Visualisation
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

def post_traitement(resultats, image_originale, caracteristiques, visualiser=False):
    """
    Applique un post-traitement aux résultats de détection d'escaliers pour
    réduire les faux positifs et améliorer la précision.
    
    Args:
        resultats (dict): Résultats de la classification d'escalier
        image_originale (numpy.ndarray): Image originale
        caracteristiques (dict): Caractéristiques extraites
        visualiser (bool): Si True, affiche les résultats
        
    Returns:
        dict: Résultats post-traités
    """
    # Si aucun escalier n'a été détecté, pas besoin de post-traitement
    if not resultats["est_escalier"]:
        return resultats
    
    # 1. Affiner les contours pour obtenir une meilleure segmentation
    masque_affine = affiner_contours(
        resultats["masque_escalier"],
        kernel_size=(5, 5),
        iterations=2,
        visualiser=visualiser
    )
    
    # 2. Vérifier la cohérence spatiale
    masque_coherent, score_coherence = verifier_coherence_spatiale(
        masque_affine,
        caracteristiques.get("lignes_horizontales"),
        image_originale.shape,
        visualiser=visualiser
    )
    
    # 3. Mettre à jour le score de détection en tenant compte de la cohérence spatiale
    score_original = resultats["score"]
    score_ajuste = (score_original * 0.7) + (score_coherence * 0.3)  # Pondération 70%-30%
    
    # 4. Réévaluer la confiance
    if score_ajuste >= 70:
        confiance = "Élevée"
        est_escalier = True
    elif score_ajuste >= 50:
        confiance = "Moyenne"
        est_escalier = True
    elif score_ajuste >= 30:
        confiance = "Faible"
        est_escalier = False  # Rejeté après post-traitement
    else:
        confiance = "Très faible"
        est_escalier = False  # Rejeté après post-traitement
    
    # Si le score de cohérence est très bas, rejeter la détection
    if score_coherence < 30:
        est_escalier = False
        confiance = "Très faible (rejeté après post-traitement)"
    
    # 5. Générer une visualisation mise à jour
    image_post_traitement = image_originale.copy()
    
    # Trouver les contours du masque cohérent pour l'affichage
    contours_masque, _ = cv2.findContours(masque_coherent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dessiner les contours sur l'image
    if est_escalier:
        cv2.drawContours(image_post_traitement, contours_masque, -1, (0, 255, 0), 2)
        
        # Trouver le plus grand contour pour le texte
        if contours_masque:
            c = max(contours_masque, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(
                image_post_traitement, 
                f"ESCALIER ({score_ajuste:.1f}%)", 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
    else:
        cv2.putText(
            image_post_traitement, 
            f"PAS D'ESCALIER ({score_ajuste:.1f}%)", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 0, 255), 
            2
        )
    
    # Visualisation
    if visualiser:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Image originale")
        plt.imshow(cv2.cvtColor(image_originale, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title(f"Classification initiale (score: {score_original:.1f}%)")
        plt.imshow(cv2.cvtColor(resultats["image_classification"], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Masque post-traité")
        plt.imshow(masque_coherent, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title(f"Classification finale (score: {score_ajuste:.1f}%, cohérence: {score_coherence:.1f}%)")
        plt.imshow(cv2.cvtColor(image_post_traitement, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les détails
        print("\n=== Résultats du post-traitement ===")
        print(f"Score initial: {score_original:.1f}%")
        print(f"Score de cohérence spatiale: {score_coherence:.1f}%")
        print(f"Score final: {score_ajuste:.1f}%")
        print(f"Statut: {'ESCALIER DÉTECTÉ' if est_escalier else 'PAS D\'ESCALIER'}")
        print(f"Confiance: {confiance}")
    
    # Mettre à jour les résultats
    resultats_post_traites = resultats.copy()
    resultats_post_traites["est_escalier"] = est_escalier
    resultats_post_traites["score"] = score_ajuste
    resultats_post_traites["score_initial"] = score_original
    resultats_post_traites["score_coherence"] = score_coherence
    resultats_post_traites["confiance"] = confiance
    resultats_post_traites["masque_escalier"] = masque_coherent
    resultats_post_traites["image_classification"] = image_post_traitement
    
    return resultats_post_traites

if __name__ == "__main__":
    # Tester le post-traitement sur une image
    try:
        from extraction_caracteristiques import detection_escaliers_complete
        
        # Chemin vers une image d'escalier
        image_path = "images/14.jpg"
        
        # Exécuter le pipeline de détection
        print("Détection d'escaliers...")
        resultats = detection_escaliers_complete(image_path, visualiser=False)
        
        if resultats:
            # Charger l'image originale
            image = cv2.imread(image_path)
            
            # Appliquer le post-traitement
            print("Application du post-traitement...")
            resultats_post_traites = post_traitement(resultats, image, {}, visualiser=True)
            
            # Sauvegarder les résultats
            cv2.imwrite("resultat_post_traitement.jpg", resultats_post_traites["image_classification"])
            cv2.imwrite("masque_post_traitement.jpg", resultats_post_traites["masque_escalier"])
            print("Résultats sauvegardés!")
    except Exception as e:
        print(f"Erreur lors du test: {e}")
