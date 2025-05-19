# extraction_texture_materiau.py
import cv2, numpy as np
from scipy.stats import skew
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from joblib import load

# ---------- 1. Couleur ----------------------------
def couleur_features(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    feats = []

    for img in (lab, hsv):
        for c in cv2.split(img):
            c_flat = c.reshape(-1)
            c_flat = c_flat[c_flat > 0]          # ignorer les zéros
            if c_flat.size == 0:                 # masque vide
                feats += [0., 0., 0.]
                continue
            mean = c_flat.mean()
            std  = c_flat.std()
            skewness = 0.0 if std < 1e-6 else skew(c_flat, bias=False)
            feats += [mean, std, skewness]

    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8],
                        [0,180, 0,256, 0,256]).flatten()

    n_pix = np.count_nonzero(hsv[:, :, 0])      # ← nb de pixels « réels »
    if n_pix == 0:
        hist_norm = np.zeros_like(hist, dtype=np.float32)  # ROI vide
    else:
        hist_norm = hist / n_pix

    return np.hstack([feats, hist_norm])

# ---------- 2. Texture ----------------------------
def texture_features(gray):
    lbp_feats = []
    for R in (1,2,3):
        lbp = local_binary_pattern(gray, 8*R, R, method='uniform')
        n_bins = int(lbp.max() + 1)
        lbp_feats += np.histogram(lbp,
                                  bins=n_bins,
                                  range=(0, n_bins))[0].tolist()
    gcm = graycomatrix(gray,
                       distances=[1,2,4],
                       angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                       symmetric=True,
                       normed=True)
    har = [graycoprops(gcm, p).mean()
           for p in ('contrast', 'dissimilarity', 'homogeneity',
                     'ASM', 'energy', 'correlation')]
    return np.hstack([lbp_feats, har])

# ---------- 3. Brillance --------------------------
def brilliance_ratio(hsv):
    v = hsv[:,:,2] / 255.0
    return np.array([np.mean(v > 0.85)])

# ---------- 4. Vecteur global ---------------------
def extraire_features_materiau(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feats = np.hstack([couleur_features(roi),
                       texture_features(gray),
                       brilliance_ratio(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))])
    return feats.reshape(1, -1)

# ---------- 5. Prédiction -------------------------
DEFAULT_MODEL = r"C:/Users/omara/Image/code/modele_materiau.joblib"

def predire_materiau(roi, modele=DEFAULT_MODEL):
    clf, scaler, label_enc = load(modele)        # (clf, scaler, LabelEncoder)
    X = scaler.transform(extraire_features_materiau(roi))
    proba = clf.predict_proba(X).max()
    return label_enc.inverse_transform(clf.predict(X))[0], proba
