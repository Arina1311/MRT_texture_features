import numpy as np
import cv2
from skimage.feature import hog

def calculate_hog_features_from_images(images):
    """Apskaičiuoja HOG vidurkį ir dispersiją per visus paveikslėlius."""

    feats = []
    for img in images:
        if img is None:
            continue
        if img.ndim == 3:  # jei spalvotas – paverčiam į pilką
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        feats.append(f)

    if not feats:  # jei nėra duomenų
        return {"mean": 0.0, "variance": 0.0}

    F = np.vstack(feats)       # [n_slices, feature_len]
    flat = F.ravel()           # viską sujungiame į vieną vektorių

    return {
        "mean": float(flat.mean()),
        "variance": float(flat.var())
    }
