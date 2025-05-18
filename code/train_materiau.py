import os, json, cv2
import base64

import numpy as np
from joblib import dump
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from extraction_texture_materiau import extraire_features_materiau

labels_dir = r"C:/Users/omara/Image/Labelised"       # <-- ton dossier
out_path   = r"C:/Users/omara/Image/code/modele_materiau.joblib"
X, y = [], []

for js in os.listdir(labels_dir):
    if not js.endswith(".json"):
        continue
    with open(os.path.join(labels_dir, js), encoding="utf-8") as f:
        data = json.load(f)

   # 1. Récupérer l’image -------------------------------------------------
    rel_name = data["imagePath"].lstrip("./\\")        # ex. 10.jpg
    img_path = os.path.join(labels_dir, rel_name)      # Labelised\10.jpg
    img_path = os.path.normpath(img_path)
    img = cv2.imread(img_path)

    if img is None:                                    # <-- retombe ici
        if data.get("imageData"):                      #   on décode base-64
            img_bytes = base64.b64decode(data["imageData"])
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Image introuvable :", img_path)
            continue

    # 2. Construire un masque rectangle à partir des deux points
    pts = np.array(data["shapes"][0]["points"], dtype=np.int32)   # shape (N, 2)

    x1, y1 = pts.min(axis=0)   # coin haut-gauche
    x2, y2 = pts.max(axis=0)   # coin bas-droit

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    roi = cv2.bitwise_and(img, img, mask=mask)

    # 3. Vecteur de descripteurs
    feats = extraire_features_materiau(roi).ravel()
    X.append(feats)

    # 4. Label matériau (normalisé en minuscules)
    y.append(data["shapes"][0]["label"].lower())

print("Exemples chargés :", len(X))
if not X:
    raise RuntimeError("Aucun échantillon – vérifie le dossier et la boucle ci-dessus.")

X = np.array(X)
le = LabelEncoder().fit(y)
y  = le.transform(y)

scaler = StandardScaler().fit(X)
Xn = scaler.transform(X)

param = {"C":[1,10,100], "gamma":["scale",0.01,0.001]}
clf = GridSearchCV(SVC(probability=True), param, cv=5).fit(Xn, y)

dump((clf.best_estimator_, scaler, le), out_path)
print("Modèle enregistré dans", out_path)
