import argparse
import json
import pathlib
from typing import Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

#  Règles et constantes (faciles à ajuster mais arbitraires)
METAL_STD_MAX = 5.0           # Écart‑type max. du canal V pour considérer un métal "lisse" pcq value c'est la luminosité, on assume que 
                                # que ça reflete plus de lumiere 
METAL_UR_MAX  = 0.15          # Ratio de motif LBP uniforme (flat) max. pour un métal
# LBP = Local Binary Pattern Pour chaque pixel on regarde un petit voisinage circulaire  on compare chaque voisin à la valeur du pixel central :
# si le voisin ≥ centre → 1 ; sinon → 0. On lit les 0/1 dans l’ordre horaire pour obtenir un mot binaire de 8 bits. 
# puis on convertit ce mot en entier : 00000000₂ = 0, 00011100₂ = 28, etc. L’image de ces entiers est la carte LBP.
#pourquoi ? on résume la texture d’une région en faisant simplement l’histogramme des valeurs LBP
#source :  https://fr.wikipedia.org/wiki/Motif_binaire_local 
WOOD_H_MIN, WOOD_H_MAX = 10, 25   # Hue moyen caractéristique du bois (brun)
CONCR_H_LOW_MAX   = 10        # seuil bas de Hue pour béton gris/bleuté là c plus les couleurs qui nous intéressent
CONCR_H_HIGH_MIN  = 140       # et seuil haut de Hue pour béton gris/jaunâtre


#  Lecture de l'image et extraction éventuelle de notre zone d'intéret (dans notre cas les marches) depuis un JSON LabelMe

def load_image_and_roi(img_path: str, json_path: Optional[str] = None) -> np.ndarray:
    """Retourne l'image ou le ROI .

    Si *json_path* est fourni, il doit pointer vers une annotation LabelMe
    contenant un polygone ou un rectangle dans `data["shapes"][0]["points"]`.
    Le plus petit rectangle englobant ces points est alors utilisé pour
    extraire la région d'intérêt.
    """

    img_path = pathlib.Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    # si pas de masquage → on renvoie l'image complète
    if json_path is None:
        return img

    # lecture de l'annotation
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        pts = np.asarray(data["shapes"][0]["points"], dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

    # Masque rectangulaire blanc sur fond noir
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    roi = cv2.bitwise_and(img, img, mask=mask)
    return roi


#  Extraction *minimaliste* de caractéristiques 
def compute_features(roi_bgr: np.ndarray) -> tuple[float, float, float]:
    """Calcule (H_mean, std(V), ratio_uniform_LBP) sur la ROI.

    * **H_mean** : moyenne du canal Hue (0‑179). Zéros ignorés pour éviter le fond noir.
    * **std(V)** : écart‑type du canal Value (0‑255).
    * **uniform_ratio** : proportion de pixels dont LBP == 0 (zone plate) pour R=1.
    """

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    valid = hsv[:, :, 2] > 0   # ignore les pixels complètement noirs
    if not np.any(valid):
        return 0.0, 0.0, 1.0   # ROI vide → valeurs par défaut

    h_mean = float(np.mean(hsv[:, :, 0][valid]))
    v_std  = float(np.std(hsv[:, :, 2][valid]))

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    uniform_ratio = float(np.sum(lbp == 0) / lbp.size)

    return h_mean, v_std, uniform_ratio



# classification gloutonne qui repose sur des regles simples de coloriage et de texture qui repose sur les motifs binaires locaux

def classify_material(roi_bgr: np.ndarray) -> str:
    """Retourne "bois", "béton", "métal" ou "inconnu" à partir des règles fixes."""

    H, Vstd, ur = compute_features(roi_bgr)

    # métal très lisse / brillant
    if Vstd < METAL_STD_MAX and ur < METAL_UR_MAX:
        return "métal"

    # bois brun
    if WOOD_H_MIN <= H <= WOOD_H_MAX:
        return "bois"

    # béton gris (Hue très bas ou très haut)
    if H < CONCR_H_LOW_MAX or H > CONCR_H_HIGH_MIN:
        return "béton"

    return "inconnu"


#  interface ligne de commande 

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Détection de matériau sans modèle ML (règles fixes).")
    parser.add_argument("image", help="Chemin de l'image à tester (BGR/RGB)")
    parser.add_argument("--json", help="Annotation LabelMe pour extraire un ROI (optionnel)")
    args = parser.parse_args()

    roi = load_image_and_roi(args.image, args.json)
    mat = classify_material(roi)
    print(mat)

if __name__ == "__main__":
    _cli()
