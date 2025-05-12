import cv2
import numpy as np
import matplotlib.pyplot as plt

def amplifier_contours_verticaux(image, facteur_amplification=2.0):
    """
    Amplifie spécifiquement les contours verticaux dans une image.
    
    Args:
        image (numpy.ndarray): Image d'entrée (gradients ou contours)
        facteur_amplification (float): Facteur par lequel amplifier les contours verticaux
        
    Returns:
        numpy.ndarray: Image avec les contours verticaux amplifiés
    """
    # Appliquer un filtre de Sobel vertical pour extraire les contours verticaux
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Conversion en valeur absolue et normalisation
    gradient_y_abs = cv2.convertScaleAbs(gradient_y)
    
    # Appliquer un filtre morphologique pour renforcer les structures verticales
    kernel_vertical = np.ones((5, 1), np.uint8)  # Noyau vertical
    contours_verticaux = cv2.morphologyEx(gradient_y_abs, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Amplifier les contours verticaux dans l'image originale
    image_amplifiee = cv2.addWeighted(image, 1.0, contours_verticaux, facteur_amplification - 1.0, 0)
    return image_amplifiee

def detection_contours(image_pretraitee, visualiser=False):
    """
    Réalise la détection des contours en utilisant l'implémentation optimisée d'OpenCV.
    Donne une importance accrue aux contours verticaux pour la détection d'escaliers.
    
    Args:
        image_pretraitee (numpy.ndarray): Image prétraitée et seuillée
        visualiser (bool): Si True, affiche les étapes intermédiaires
        
    Returns:
        contours (numpy.ndarray): Image des contours détectés avec emphase sur les verticaux
    """
    # 1 & 2. Calcul des gradients avec Sobel
    # Gradient horizontal (dérivée selon x)
    gradient_x = cv2.Sobel(image_pretraitee, cv2.CV_64F, 1, 0, ksize=3)
    # Gradient vertical (dérivée selon y) - utilisé pour les contours horizontaux
    gradient_y = cv2.Sobel(image_pretraitee, cv2.CV_64F, 0, 1, ksize=3)
    
    # Donner plus de poids au gradient y pour accentuer les contours horizontaux
    # (qui correspondent aux marches d'escalier horizontales)
    poids_x = 1.0  # Poids normal pour les contours verticaux
    poids_y = 2.5  # Poids plus élevé pour les contours horizontaux (qui détectent les marches)
    
    # Magnitude du gradient pondérée
    magnitude = np.sqrt((poids_x * gradient_x)**2 + (poids_y * gradient_y)**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Direction du gradient
    direction = np.arctan2(gradient_y, gradient_x)
    
    # Amplifier spécifiquement les contours verticaux
    magnitude_amplifiee = amplifier_contours_verticaux(magnitude, facteur_amplification=1.8)
    
    # Calcul des contours avec l'implémentation optimisée de Canny d'OpenCV
    # Nous utilisons des seuils adaptés à une image déjà seuillée
    seuil_bas = 30  # Ces valeurs peuvent être ajustées
    seuil_haut = 100
    contours_opencv = cv2.Canny(magnitude_amplifiee.astype(np.uint8), seuil_bas, seuil_haut)
    
    # Renforcement morphologique des contours verticaux
    kernel_vertical = np.ones((7, 1), np.uint8)  # Noyau vertical plus grand
    contours_verticaux = cv2.morphologyEx(contours_opencv, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Combiner les contours originaux et les contours verticaux renforcés
    contours_finaux = cv2.addWeighted(contours_opencv, 0.7, contours_verticaux, 0.3, 0)
    
    # Visualisation des étapes si demandé
    if visualiser:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("Image prétraitée (seuillée)")
        plt.imshow(image_pretraitee, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("Gradient horizontal (Sobel X)")
        plt.imshow(cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("Gradient vertical (Sobel Y)")
        plt.imshow(cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("Magnitude du gradient amplifiée")
        plt.imshow(magnitude_amplifiee, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("Contours avec Canny")
        plt.imshow(contours_opencv, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.title("Contours finaux (verticaux amplifiés)")
        plt.imshow(contours_finaux, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return contours_finaux

def test_detection_contours():
    """Fonction de test pour la détection des contours"""
    try:
        # Utiliser le module de prétraitement existant
        from pretraitement import pretraitement_image
        
        # Chemin vers une image d'escalier (à remplacer par votre propre chemin)
        image_path = "images/4.jpg"
        
        # Prétraitement de l'image
        print("Application du prétraitement...")
        image_pretraitee = pretraitement_image(image_path, visualiser=False)
        print("Prétraitement réussi!")
        # Appliquer la détection des contours avec visualisation
        print("Détection des contours...")
        contours = detection_contours(image_pretraitee, visualiser=True)
        print("Détection des contours réussie!")
        
        # Sauvegarde de l'image des contours
        cv2.imwrite("contours_detectes.jpg", contours)
        print("Image des contours sauvegardée sous 'contours_detectes.jpg'")
        
    except ImportError:
        print("Erreur: Le module 'pretraitement.py' n'a pas été trouvé.")
        print("Assurez-vous que le fichier 'pretraitement.py' est dans le même répertoire que ce script.")
    except Exception as e:
        print(f"Erreur lors de la détection des contours: {e}")

if __name__ == "__main__":
    test_detection_contours()