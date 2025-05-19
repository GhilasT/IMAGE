import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from extraction_caracteristiques import detection_escaliers_complete
from tqdm import tqdm

def charger_label_json(chemin_json):
    """
    Charge un fichier JSON de LabelMe et extrait le type d'escalier annoté
    ainsi que le chemin vers l'image associée.
    
    Args:
        chemin_json (str): Chemin vers le fichier JSON
    
    Returns:
        tuple: (type_escalier, chemin_image) ou (None, None) si non trouvé
    """
    try:
        with open(chemin_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraire le chemin de l'image
        chemin_image_relatif = data.get('imagePath')
        if not chemin_image_relatif:
            print(f"Pas de chemin d'image trouvé dans {chemin_json}")
            return None, None
            
        # Convertir le chemin en absolu, en tenant compte du format "../images/nom.jpg"
        dossier_json = os.path.dirname(chemin_json)
        
        # Si le chemin d'image commence par "../", on remonte d'un niveau supplémentaire
        if chemin_image_relatif.startswith("../"):
            chemin_image = os.path.normpath(os.path.join(dossier_json, chemin_image_relatif))
        else:
            # Sinon, on traite comme un chemin relatif normal
            chemin_image = os.path.normpath(os.path.join(dossier_json, chemin_image_relatif))
        
        # Rechercher les labels "Droit" ou "Tournant" dans les shapes
        for shape in data.get('shapes', []):
            label = shape.get('label', '').lower()
            if label in ['droit', 'tournant']:
                return label, chemin_image
                
        return None, chemin_image  # Label non trouvé mais image trouvée
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {chemin_json}: {e}")
        return None, None

def creer_matrice_confusion(y_true, y_pred, classes):
    """
    Crée une matrice de confusion manuellement sans utiliser sklearn.
    
    Args:
        y_true: Liste des labels réels
        y_pred: Liste des labels prédits
        classes: Liste des classes possibles
    
    Returns:
        np.array: Matrice de confusion
    """
    n_classes = len(classes)
    matrice = np.zeros((n_classes, n_classes), dtype=int)
    
    # Créer un dictionnaire pour mapper les noms de classes à des indices
    classe_vers_index = {classe: i for i, classe in enumerate(classes)}
    
    # Remplir la matrice de confusion
    for vrai, pred in zip(y_true, y_pred):
        i = classe_vers_index[vrai]
        j = classe_vers_index[pred]
        matrice[i, j] += 1
    
    return matrice

def afficher_matrice_confusion(cm, classes, titre='Matrice de confusion'):
    """
    Affiche une matrice de confusion sans utiliser sklearn.
    
    Args:
        cm: Matrice de confusion
        classes: Liste des noms de classes
        titre: Titre du graphique
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(titre)
    plt.colorbar()
    
    # Ajouter les labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Ajouter les valeurs dans les cellules
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # On etiquette les valeurs
    # Considérer la première classe (Droit) comme la classe positive
    if len(classes) == 2:
        plt.text(0, 0, f"TP: {cm[0, 0]}", ha='center', va='center', 
                fontweight='bold', bbox=dict(facecolor='none', edgecolor='black'))
        
        plt.text(1, 0, f"FN: {cm[0, 1]}", ha='center', va='center', 
                color="white" if cm[0, 1] > thresh else "black", 
                fontweight='bold', bbox=dict(facecolor='none', edgecolor='black'))
        
        plt.text(0, 1, f"FP: {cm[1, 0]}", ha='center', va='center', 
                color="white" if cm[1, 0] > thresh else "black", 
                fontweight='bold', bbox=dict(facecolor='none', edgecolor='black'))
        
        plt.text(1, 1, f"TN: {cm[1, 1]}", ha='center', va='center', 
                color="white" if cm[1, 1] > thresh else "black", 
                fontweight='bold', bbox=dict(facecolor='none', edgecolor='black'))
    
    plt.tight_layout()
    plt.ylabel('Vrai label')
    plt.xlabel('Label prédit')
    plt.savefig('matrice_confusion.png')
    plt.show()

def evaluer_performances(dossier_labels="../Labelised_stairs_direction/", visualiser=False):
    """
    Évalue les performances du détecteur d'escaliers en calculant la matrice de confusion.
    
    Args:
        dossier_labels (str): Chemin vers le dossier contenant les JSON labelisés
        visualiser (bool): Si True, affiche des exemples de détection
    
    Returns:
        dict: Résultat de l'évaluation
    """
    print(f"Évaluation des performances en utilisant les données dans: {dossier_labels}")
    
    if not os.path.exists(dossier_labels):
        raise FileNotFoundError(f"Le dossier {dossier_labels} n'existe pas")
    
    # Recupération de tous les fichiers json
    fichiers_json = [f for f in os.listdir(dossier_labels) if f.endswith('.json')]
    print(f"Nombre de fichiers JSON trouvés: {len(fichiers_json)}")
    
    # Collecter les labels et les prédictions
    resultats = []
    
    # On parcours tout les fichiers JSON
    for json_file in tqdm(fichiers_json, desc="Traitement des images"):
        chemin_json = os.path.join(dossier_labels, json_file)
        
        label_reel, chemin_image = charger_label_json(chemin_json)
        
        if label_reel is None:
            print(f"Aucun label pertinent trouvé dans {json_file}, ignoré.")
            continue
        
        if chemin_image is None or not os.path.exists(chemin_image):
            print(f"Image non trouvée pour {json_file}: {chemin_image}")
            continue
        
        try:
            # Exécuter la détection d'escalier
            resultat_detection = detection_escaliers_complete(chemin_image, visualiser=False)
            
            # Extraire la prédiction
            if resultat_detection and "type_escalier" in resultat_detection:
                prediction = resultat_detection["type_escalier"]
                
                # Stocker les résultats
                resultats.append({
                    "fichier": os.path.basename(chemin_image),
                    "label_reel": label_reel,
                    "prediction": prediction,
                    "image_path": chemin_image,
                    "resultat_complet": resultat_detection
                })
                
                # Afficher quelques exemples de détection si flag visualiser en vrai
                if visualiser and len(resultats) % 5 == 0:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(resultat_detection["image_classification"], cv2.COLOR_BGR2RGB))
                    plt.title(f"Réel: {label_reel.upper()} - Prédit: {prediction.upper()}")
                    plt.axis('off')
                    plt.show()
                
            else:
                print(f"Échec de la détection pour {chemin_image}")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {chemin_image}: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyser les résultats si des données ont été collectées
    if resultats:
        df_resultats = pd.DataFrame(resultats)
        
        # Calculer la matrice de confusion manuellement
        classes = ['droit', 'tournant']
        y_true = df_resultats['label_reel'].values
        y_pred = df_resultats['prediction'].values
        
        cm = creer_matrice_confusion(y_true, y_pred, classes)
        
        # Afficher la matrice de confusion
        afficher_matrice_confusion(cm, ['Droit', 'Tournant'], 
                               titre='Matrice de confusion - Détection du type d\'escalier')
        
        # Extraire les valeurs pour les métriques afin de calculer les évaluations
        # - TP = escalier droit correctement prédit droit (cm[0,0])
        # - TN = escalier tournant correctement prédit tournant (cm[1,1])
        # - FP = escalier tournant incorrectement prédit droit (cm[1,0])
        # - FN = escalier droit incorrectement prédit tournant (cm[0,1])
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]
        
        # Calculer les métriques de performance
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Précision des escaliers droits
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # Rappel des escaliers droits
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n=== Métriques de performance (pour les escaliers DROITS) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        # Afficher des exemples d'erreurs
        erreurs = df_resultats[df_resultats['label_reel'] != df_resultats['prediction']]
        if not erreurs.empty and visualiser:
            print(f"\nAffichage de {min(3, len(erreurs))} exemples d'erreurs:")
            for i, (_, row) in enumerate(erreurs.head(3).iterrows()):
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(row['resultat_complet']["image_classification"], cv2.COLOR_BGR2RGB))
                plt.title(f"ERREUR - Réel: {row['label_reel'].upper()} - Prédit: {row['prediction'].upper()}")
                plt.axis('off')
                plt.show()
        
        return {
            'matrice_confusion': cm,
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1_score,
            'resultats_detailles': df_resultats
        }
    else:
        print("Aucun résultat obtenu pour l'évaluation.")
        return None

def generer_rapport(resultats, chemin_sortie="rapport_performance.csv"):
    """
    Génère un rapport détaillé des résultats de l'évaluation.
    
    Args:
        resultats (dict): Résultats de l'évaluation
        chemin_sortie (str): Chemin où sauvegarder le rapport CSV
    """
    if resultats is None or 'resultats_detailles' not in resultats:
        print("Aucun résultat à exporter.")
        return
        
    # Exporter les résultats détaillés
    df_resultats = resultats['resultats_detailles']
    df_resultats['correct'] = df_resultats['label_reel'] == df_resultats['prediction']
    
    # Ajout des métriques globales
    df_metriques = pd.DataFrame([{
        'accuracy': resultats['accuracy'],
        'precision': resultats['precision'],
        'recall': resultats['recall'],
        'f1_score': resultats['f1_score']
    }])
    
    # Exporter les deux DataFrames
    df_resultats.to_csv(chemin_sortie, index=False)
    df_metriques.to_csv("metriques_" + chemin_sortie, index=False)
    
    print(f"Rapport détaillé sauvegardé dans {chemin_sortie}")
    print(f"Métriques globales sauvegardées dans metriques_{chemin_sortie}")

if __name__ == "__main__":
    try:
        # Exécuter l'évaluation des performances
        resultats = evaluer_performances(
            dossier_labels="../Labelised_stairs_direction/", 
            visualiser=True
        )
        
        # Générer un rapport des résultats
        if resultats:
            generer_rapport(resultats)
            
    except Exception as e:
        import traceback
        print(f"Erreur lors de l'évaluation des performances: {e}")
        traceback.print_exc()
