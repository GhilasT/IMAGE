import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from pretraitement import pretraitement_image
from detection_contours import detection_contours

def extraction_lignes_simples(image_contours, visualiser=False):
    """
    Extrait simplement les lignes droites à partir d'une image de contours en utilisant
    la transformée de Hough, puis filtre les lignes selon leur orientation.
    
    Args:
        image_contours (numpy.ndarray): Image des contours détectés
        visualiser (bool): Si True, affiche les résultats
        
    Returns:
        dict: Dictionnaire contenant les lignes horizontales et verticales
    """
    # Créer une image en couleur pour l'affichage des résultats
    image_resultats = cv2.cvtColor(image_contours, cv2.COLOR_GRAY2BGR) if len(image_contours.shape) == 2 else image_contours.copy()
    
    # On applique la trasformée de Hough pour détécler les lignes
    lignes = cv2.HoughLinesP(
        image_contours,      
        rho=1,              
        theta=np.pi/180,     
        threshold=50,        
        minLineLength=50,    
        maxLineGap=10       
    )
    
    if lignes is None:
        print("Aucune ligne détectée!")
        return None
    
    lignes_horizontales = []
    lignes_verticales = []
    
    # Image pour afficher séparément les lignes horizontales et verticales
    image_horizontales = image_resultats.copy()
    image_verticales = image_resultats.copy()
    
    for ligne in lignes:
        x1, y1, x2, y2 = ligne[0]
        
        # On calcule l'angle de la ligne
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        if angle < 30 or angle > 150:  # On applique une tolérance de 30 degré
            lignes_horizontales.append(ligne)
            cv2.line(image_resultats, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(image_horizontales, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        elif 60 < angle < 120:
            lignes_verticales.append(ligne)
            cv2.line(image_resultats, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(image_verticales, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Visualisation des résultats si le flag visualiser est a True
    if visualiser:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Image des contours")
        plt.imshow(image_contours, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title(f"Toutes les lignes (H: {len(lignes_horizontales)}, V: {len(lignes_verticales)})")
        plt.imshow(cv2.cvtColor(image_resultats, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title(f"Lignes horizontales ({len(lignes_horizontales)})")
        plt.imshow(cv2.cvtColor(image_horizontales, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title(f"Lignes verticales ({len(lignes_verticales)})")
        plt.imshow(cv2.cvtColor(image_verticales, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Résultats simples
    resultats = {
        "lignes_horizontales": lignes_horizontales,
        "lignes_verticales": lignes_verticales,
        "image_resultats": image_resultats,
        "image_horizontales": image_horizontales,
        "image_verticales": image_verticales
    }
    
    return resultats

def detection_escaliers_simplifiee(image_path, visualiser=False):
    """
    Pipeline simplifié pour l'extraction des lignes d'escaliers:
    1. Prétraitement de l'image
    2. Détection des contours
    3. Extraction des lignes et filtrage par orientation
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        visualiser (bool): Si True, affiche les résultats de chaque étape
    
    Returns:
        dict: Résultats contenant les lignes et images
    """
    print("1. Prétraitement de l'image...")
    image_pretraitee = pretraitement_image(image_path, visualiser=visualiser)
    
    print("2. Détection des contours...")
    contours = detection_contours(image_pretraitee, visualiser=visualiser)
    
    print("3. Extraction des lignes droites...")
    resultats = extraction_lignes_simples(contours, visualiser=visualiser)
    
    if resultats:
        print("\nExtraction terminée!")
        print(f"Lignes horizontales détectées: {len(resultats['lignes_horizontales'])}")
        print(f"Lignes verticales détectées: {len(resultats['lignes_verticales'])}")
        
        # Sauvegarder les images finales
        cv2.imwrite("lignes_filtrees.jpg", resultats['image_resultats'])
        cv2.imwrite("lignes_horizontales.jpg", resultats['image_horizontales'])
        cv2.imwrite("lignes_verticales.jpg", resultats['image_verticales'])
        print("Images des lignes sauvegardées!")
    
    return resultats

if __name__ == "__main__":
    image_path = "../images/4.jpg"
    
    try:
        # On execute tout le processus de traitement
        resultats = detection_escaliers_simplifiee(image_path, visualiser=True)
    except Exception as e:
        print(f"Erreur lors de l'extraction des lignes: {e}")