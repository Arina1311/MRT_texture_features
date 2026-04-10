import os
from typing import List, Tuple, Dict
import cv2
import numpy as np
from cv2.typing import MatLike

BG_THRESHOLD = 0        # > 0 ≙ miokardo pikseliai po spalvų nuėmimo
MIN_BBOX_AREA = 1_000   # atmetam labai mažas dėžutes

APICAL_AREA_DROP = 0.70  # 70% riba apačios (apex) pradžiai
BASAL_TOLERANCE = 0.97   # 97% riba viršaus (basal) pabaigai

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def detect_grid_boxes(grid: MatLike) -> List[Tuple[int, int, int, int]]:
    """
    Randa trumpųjų ašių skilteles viename „grid“ vaizde ir grąžina dėžučių sąrašą (y, x, w, h),
    suklasterintą eilutėmis ir surikiuotą iš kairės į dešinę (deterministiškai).
    """
    mask = (grid > BG_THRESHOLD).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_BBOX_AREA:
            boxes.append((y, x, w, h))

    if not boxes:
        return []

    # Vidutinis aukštis -> eilutės tolerancijai
    avg_h = float(np.mean([b[3] for b in boxes]))
    tol = avg_h / 2.0

    rows: List[List[Tuple[int, int, int, int]]] = []
    # Rikiuojam stabiliai: pirmiausia pagal y-centrą, tada x-centrą, o jei lygūs – pagal lauką
    for box in sorted(
        boxes,
        key=lambda b: (b[0] + b[3] / 2.0, b[1] + b[2] / 2.0, b[1], b[0], b[2], b[3]),
    ):
        cy = box[0] + box[3] / 2.0
        placed = False
        for row in rows:
            row_cy = sum(r[0] + r[3] / 2.0 for r in row) / len(row)
            if abs(cy - row_cy) <= tol:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # Iš kiekvienos eilės – rikiuojame kairėn->dešinėn (pagal x-centrą)
    sorted_boxes: List[Tuple[int, int, int, int]] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda b: (b[1] + b[2] / 2.0, b[1], b[0], b[2], b[3]))
        sorted_boxes.extend(row_sorted)

    return sorted_boxes

# Miokardas = > BG_THRESHOLD
def myocardium_only(img: np.ndarray) -> np.ndarray:
    return (img > BG_THRESHOLD).astype(np.uint8)

# Ertmės (kairiojo skilvelio) plotas ROI viduje (apytiksliai)
def cavity_area(img: np.ndarray) -> int:
    m = myocardium_only(img)
    if m.sum() == 0:
        return 0
    ys, xs = np.where(m)
    roi = img[min(ys): max(ys) + 1, min(xs): max(xs) + 1]
    return int((roi <= BG_THRESHOLD).sum())

def split_regions(slices: List[np.ndarray]) -> Dict[str, List[int]]:
    """
    Skirsto skilteles į top/mid/bottom pagal ertmės plotą (cavity_area):
      - didžiausia ertmė ~ mid,
      - iki ~97% nuo max laikome dar „top“,
      - ~70% nuo max ir mažiau laikome „bottom“.
    Jei ribos neaiškios — fallback į tris lygius segmentus.
    """
    n = len(slices)
    if n < 3:
        return {
            "top": list(range(min(1, n))),
            "mid": list(range(1, min(2, n))),
            "bottom": list(range(2, n)),
        }

    areas = [cavity_area(s) for s in slices]
    mid_idx = int(np.argmax(areas))
    max_a = areas[mid_idx]
    thr_basal = max_a * BASAL_TOLERANCE
    thr_apical = max_a * APICAL_AREA_DROP

    top_start = 0
    bottom_start = n

    # Nustatome viršutinės srities pabaigą (nuo max žemyn)
    for i in range(mid_idx + 1, n):
        if areas[i] < thr_basal:
            top_start = i
            break

    # Nustatome apatinės srities pradžią
    for i in range(top_start, n):
        if areas[i] < thr_apical:
            bottom_start = i
            break

    if bottom_start == n or top_start == bottom_start:
        # fallback: 3 maždaug lygios dalys
        t = n // 3
        top_count = t
        bottom_count = t
        mid_count = n - (top_count + bottom_count)
        return {
            "top": list(range(0, top_count)),
            "mid": list(range(top_count, top_count + mid_count)),
            "bottom": list(range(top_count + mid_count, n)),
        }

    return {
        "top": list(range(0, top_start)),
        "mid": list(range(top_start, bottom_start)),
        "bottom": list(range(bottom_start, n)),
    }
