import cv2
import numpy as np
import matplotlib.pyplot as plt

def pretraitement_image(image_path, visualiser=False):
    """
    Réalise le prétraitement d'une image pour la détection d'escaliers.
    
    Étapes:
    1. Conversion en niveaux de gris
    2. Application d'un filtre gaussien pour réduire le bruit
    3. Égalisation d'histogramme pour améliorer le contraste
    
    Args:
        image_path (str): Chemin vers l'image à traiter
        visualiser (bool): Si True, affiche les étapes intermédiaires
        
    Returns:
        image_traitee (numpy.ndarray): Image prétraitée
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")

    # On applique tout le pre traitement
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_lissee = cv2.GaussianBlur(image_gris, (5, 5), 0) # Flou Gaussien
    
    image_egalisee = cv2.equalizeHist(image_lissee) # On egalise l'histogramme
    
    # Ici on n'applique pas Otsu
    image_traitee = image_egalisee
    
    # Visualisation des étapes si demandé
    if visualiser:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Image originale")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Image en niveaux de gris")
        plt.imshow(image_gris, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Image après filtre gaussien")
        plt.imshow(image_lissee, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title("Image prétraitée (après égalisation d'histogramme)")
        plt.imshow(image_traitee, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Histogramme avant égalisation")
        plt.hist(image_lissee.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        
        plt.subplot(1, 2, 2)
        plt.title("Histogramme après égalisation")
        plt.hist(image_egalisee.flatten(), 256, [0, 256], color='b')
        plt.xlim([0, 256])
        
        plt.tight_layout()
        plt.show()
    
    return image_traitee

def test_pretraitement():
    """Fonction de test pour le prétraitement"""
    image_path = "../images/14.jpg"
    
    try:
        # Appliquer le prétraitement avec visualisation
        image_traitee = pretraitement_image(image_path, visualiser=True)
        print("Prétraitement réussi!")
        
        # Sauvegarde de l'image prétraitée
        cv2.imwrite("image_pretraitee.jpg", image_traitee)
        print("Image prétraitée sauvegardée sous 'image_pretraitee.jpg'")
        
    except Exception as e:
        print(f"Erreur lors du prétraitement: {e}")

if __name__ == "__main__":
    test_pretraitement()