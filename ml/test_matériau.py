import argparse
import cv2
import sys

# Assure that the local package path is on sys.path so Python can import the module
sys.path.append('.')

try:
    from extraction_texture_materiau import predire_materiau, DEFAULT_MODEL
except ImportError as e:
    raise SystemExit("Impossible d\'importer extraction_texture_materiau.py : assure‑toi que le fichier se trouve dans le même dossier que ce script.\nDétails : " + str(e))


def main() -> None:
    """Script de test pour la fonction predire_materiau.

    Usage :
        python test_materiau.py chemin/vers/roi.jpg [--model chemin/vers/modele.joblib]

    Le script :
    1. charge l\'image ROI passée en argument (format BGR, comme lu par cv2).
    2. appelle predire_materiau pour obtenir l\'étiquette et la probabilité.
    3. affiche le résultat sur la sortie standard.
    """

    parser = argparse.ArgumentParser(
        description="Teste la prédiction du matériau sur une image ROI.")
    parser.add_argument(
        'image',
        help='Chemin vers l\'image ROI (l\'image doit déjà être recadrée sur la zone d\'intérêt).')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help="Chemin vers le fichier .joblib du modèle (par défaut : %(default)s).")

    args = parser.parse_args()

    # Lecture de l'image
    roi = cv2.imread(args.image)
    if roi is None:
        raise SystemExit(f"Erreur : impossible de lire l\'image '{args.image}'.")

    # Prédiction
    try:
        label, proba = predire_materiau(roi, args.model)
    except FileNotFoundError as e:
        raise SystemExit(f"Fichier modèle introuvable : {e}")
    except Exception as e:
        raise SystemExit(f"Une erreur est survenue pendant la prédiction : {e}")

    # Affichage
    print(f"Matériau prédit : {label} (confiance : {proba*100:.1f}%)")


if __name__ == '__main__':
    main()
