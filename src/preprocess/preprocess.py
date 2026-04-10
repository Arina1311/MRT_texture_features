import json
import os
import cv2
from cv2.typing import MatLike
import numpy as np
from typing import List, Tuple
from pathlib import Path
import hashlib

from .slice_split import detect_grid_boxes, ensure_dir, split_regions

# ---- KELIAI ----------------------------------------------------

circles_root_path = "/Users/arinaperzu/Desktop/MRT2/data/CIRCLES"
path_to_phase = "cine/PNG/"
extension = ".png"

prepared_root_path = "data/prepared/"

# Paspartinimui (neįtakoja deterministiškumo)
try:
    cv2.setNumThreads(os.cpu_count() or 0)
except Exception:
    pass


# ---- NAUDINGOS PAGALBINĖS --------------------------------------

def md5_file(path: str) -> str:
    """Apskaičiuoja failo MD5 (naudinga validacijai, jei prireiktų)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, obj: dict) -> None:
    """JSON rašymas atominiu būdu (į *.tmp ir po to os.replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)


def save_slice_files(phase_dir: str, s_idx: int, arr: np.ndarray) -> None:
    """Išsaugo vieną skiltelę į NPY (greitam skaitymui) ir PNG (vizualiai)."""
    np.save(os.path.join(phase_dir, f"S{s_idx}.npy"), arr)
    cv2.imwrite(os.path.join(phase_dir, f"S{s_idx}.png"), arr)


def load_slice_file(phase_dir: str, s_idx: int) -> MatLike:
    """Pirmiausia bando skaityti NPY (greita), jei nėra – PNG (vizualus)."""
    npy = os.path.join(phase_dir, f"S{s_idx}.npy")
    if os.path.exists(npy):
        return np.load(npy)
    return cv2.imread(os.path.join(phase_dir, f"S{s_idx}.png"), cv2.IMREAD_GRAYSCALE)


# ---- KAUKIŲ LOGIKA ---------------------------------------------

def get_content(
    content: MatLike, mask: MatLike, inner: bool, include_border: bool
) -> MatLike:
    """
    Iš kontūrų sudaroma pilna kaukė ir su ja:
      - jei inner=True -> paliekamas turinys kaukės VIDUJE
      - jei inner=False -> paliekamas turinys kaukės IŠORĖJE
    Jei include_border=False, papildomai pašalinama spalvota sritis (mask).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_mask = np.zeros_like(mask)
    cv2.drawContours(circle_mask, contours, -1, 255, thickness=cv2.FILLED)  # type: ignore

    content = cv2.bitwise_and(
        content, content, mask=circle_mask if inner else cv2.bitwise_not(circle_mask)
    )
    if include_border:
        return content

    return cv2.bitwise_and(content, content, mask=cv2.bitwise_not(mask))


def remove_color(content: MatLike, mask: MatLike) -> MatLike:
    """Pašalina (nunulina) sritį nurodytą maskėje iš 'content'."""
    return cv2.bitwise_and(content, content, mask=cv2.bitwise_not(mask))


# ---- VIENOS FAZĖS APDOROJIMAS ---------------------------------

def process_image(image_path: str, output_path: str, *, force_recompute: bool = True) -> MatLike:
    """
    Apdoroja vieną fazės vaizdą:
      - jei force_recompute=False ir rezultatas jau egzistuoja -> skaito GRAY iš disko (cache)
      - kitu atveju -> atlieka spalvų nuėmimą, grąžina ir įrašo GRAY
    GRĄŽINA: GRAY (MatLike)
    """
    if (not force_recompute) and os.path.exists(output_path):
        return cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)

    # Pilkos kaukė (kur kanalai lygūs)
    gray_mask = np.uint8((g == b) & (b == r)) * 255
    # Žalios kaukė (g > r ir g > b)
    green_mask = np.uint8(np.less(b, g) & np.less(r, g)) * 255

    # Raudonos pašalinimui (range + santykiai)
    lower_red = np.array([0, 0, 110])     # B,G,R
    upper_red = np.array([100, 80, 255])
    red_range = cv2.bitwise_and(
        cv2.bitwise_not(gray_mask), cv2.inRange(image, lower_red, upper_red)
    )
    red_mask_base = np.uint8((g == b) & np.less(g, r)) * 255
    red_mask = cv2.bitwise_or(red_mask_base, red_range)

    # Rožinės pašalinimui
    lower_pink = np.array([80, 0, 100])
    upper_pink = np.array([255, 60, 255])
    pink_range = cv2.bitwise_and(
        cv2.bitwise_not(gray_mask), cv2.inRange(image, lower_pink, upper_pink)
    )
    pink_base_mask = np.uint8((r == b) & np.greater(r, g)) * 255
    pink_mask = cv2.bitwise_or(pink_base_mask, pink_range)

    # Paliekam turinį žaliame žiede, tarp žalio ir raudono, ir pašalinam rožinę
    inside_green_content = get_content(image, green_mask, inner=True, include_border=False)
    between_green_and_red_content = get_content(inside_green_content, red_mask, inner=False, include_border=True)
    final_image = remove_color(between_green_and_red_content, pink_mask)

    # KONVERTUOJAM IR ĮRAŠOM GRAY (deterministiška reprezentacija)
    final_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_gray)

    return final_gray


# ---- PACIENTO FAZĖ + APDOROJIMAS -------------------------------

def process_image_by_patient_and_phase(patient: str, phase: int, *, force_recompute: bool = True) -> MatLike:
    """
    Pagal pacientą ir fazę suranda atitinkamą failą (DET. tvarka),
    apdoroja ir grąžina GRAY.
    """
    patient_dir = os.path.join(circles_root_path, patient, path_to_phase)
    phase_files = sorted(  # rikiavimas užtikrina pastovų pasirinkimą
        f for f in os.listdir(patient_dir) if f.endswith(f"_{phase}{extension}")
    )

    if not phase_files:
        raise FileNotFoundError(
            f"Nerastas atitinkamas failas pacientui {patient} ir fazei {phase}"
        )

    image_path = os.path.join(patient_dir, phase_files[0])
    prepared_image_path = os.path.join(prepared_root_path, patient, f"P{phase}{extension}")
    return process_image(image_path, prepared_image_path, force_recompute=force_recompute)


# ---- VISŲ FAZIŲ APDOROJIMAS PACIENTUI --------------------------

def process_images_by_patient(
    patient: str, start_phase: int, end_phase: int, *, force_recompute: bool = True
) -> Tuple[List[MatLike], List[MatLike], List[MatLike], List[MatLike]]:
    """
    Apdoroja fazes [start_phase, end_phase).
    Grąžina:
      - global_images: visų fazių apdoroti GRAY vaizdai
      - top/mid/bottom: pirmos fazės skiltelių vaizdai pagal regionus
    """
    global_images: List[MatLike] = []
    for i in range(start_phase, end_phase):
        image = process_image_by_patient_and_phase(patient, i, force_recompute=force_recompute)
        global_images.append(image)

    top_images, mid_images, bottom_images = slice_by_region(
        global_images, patient, start_phase, force_recompute=force_recompute
    )

    return global_images, top_images, mid_images, bottom_images


# ---- PACIENTŲ SĄRAŠAS (DET. TVARKA) ----------------------------

def get_patient_list(root_path: str) -> List[str]:
    """Grąžina paciento katalogų sąrašą rikiuota tvarka (deterministiška)."""
    return [
        f
        for f in sorted(os.listdir(root_path))
        if not f.startswith(".") and os.path.isdir(os.path.join(root_path, f))
    ]


# ---- KROPINIMAS IKI 192x192 ------------------------------------

CROP_SIZE = 192
HALF = CROP_SIZE // 2

def crop_to_192(grid: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Iškerta 192×192 centrą pagal nurodytą dėžutę (box).
    Jei trūksta pikselių – pridengia nuliais (padding).
    """
    H, W = grid.shape
    y, x, w, h = box
    cy, cx = y + h // 2, x + w // 2

    top = max(cy - HALF, 0)
    left = max(cx - HALF, 0)
    bottom = min(top + CROP_SIZE, H)
    right = min(left + CROP_SIZE, W)

    if bottom - top < CROP_SIZE:
        top = max(bottom - CROP_SIZE, 0)
    if right - left < CROP_SIZE:
        left = max(right - CROP_SIZE, 0)

    crop = grid[top:bottom, left:right]

    if crop.shape != (CROP_SIZE, CROP_SIZE):
        padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=grid.dtype)
        ph, pw = crop.shape
        padded[:ph, :pw] = crop
        crop = padded

    return crop


# ---- SKIRSTYMAS Į REGIONUS (SU CACHE) --------------------------

def slice_by_region(
    grids: List[np.ndarray], patient: str, start_phase: int, *, force_recompute: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Sukarpo visas fazes į skilteles ir pirmos fazės skilteles suskirsto į regionus.

    Struktūra diske:
      data/prepared/<patient>/
        ├─ split_info.json   (išsaugota top/mid/bottom skiltelių numeracija)
        └─ P<phase>/
            ├─ S<i>.npy / S<i>.png (192×192 — <i> = original_index + 1)

    Logika:
      - visose fazėse DET. aptinkame dėžutes ir parenkame bendrą skiltelių aibę,
      - jei force_recompute=False ir visi atitinkami failai YRA -> nerašome iš naujo, tiesiog skaitome,
      - VISADA regionus skaičiuojame pagal einamo paleidimo pirmą fazę,
      - split_info.json perrašomas atominiu būdu.
    """
    # 1) Dėžučių paieška kiekvienoje fazėje ir referencinės fazės parinkimas
    all_boxes = [detect_grid_boxes(g) for g in grids]
    ref_idx = int(np.argmax([len(b) for b in all_boxes]))  # fazė su tankiausiu grid
    ref_boxes = all_boxes[ref_idx]

    # 2) Pasiliekam tik tuos indeksus, kurie egzistuoja visose fazėse
    keep_mask = np.ones(len(ref_boxes), dtype=bool)
    for phase_boxes in all_boxes:
        phase_mask = np.zeros(len(ref_boxes), dtype=bool)
        for pb in phase_boxes:
            cy, cx = pb[0] + pb[2] // 2, pb[1] + pb[3] // 2
            for idx, rb in enumerate(ref_boxes):
                ry, rx = rb[0] + rb[2] // 2, rb[1] + rb[3] // 2
                if abs(cy - ry) < rb[2] and abs(cx - rx) < rb[3]:
                    phase_mask[idx] = True
                    break
        keep_mask &= phase_mask

    valid_orig_idx = [i for i, k in enumerate(keep_mask) if k]
    if not valid_orig_idx:
        raise RuntimeError(f"{patient}: nėra bendrų skiltelių visoms fazėms.")
    saved_numbers = [idx + 1 for idx in valid_orig_idx]

    # 3) Bandome naudoti cache (jei leidžiama) – jokio rašymo, tik skaitymas
    if not force_recompute:
        cache_ok = True
        for phase_no in range(start_phase, start_phase + len(grids)):
            phase_dir = os.path.join(prepared_root_path, patient, f"P{phase_no}")
            if not os.path.exists(phase_dir):
                cache_ok = False
                break
            for s in saved_numbers:
                if not (os.path.exists(os.path.join(phase_dir, f"S{s}.npy"))
                        or os.path.exists(os.path.join(phase_dir, f"S{s}.png"))):
                    cache_ok = False
                    break
            if not cache_ok:
                break

        if cache_ok:
            # PIRMOS FAZĖS skilteles perskaitome (greitai iš .npy jei yra)
            first_phase_dir = os.path.join(prepared_root_path, patient, f"P{start_phase}")
            first_phase_slice_imgs = [load_slice_file(first_phase_dir, s) for s in sorted(saved_numbers)]
            # Regionų skaičiavimas visada iš naujo (deterministiška ir pigu)
            regions_tmp = split_regions(first_phase_slice_imgs)

            top_images = [first_phase_slice_imgs[i] for i in sorted(regions_tmp.get("top", []))]
            mid_images = [first_phase_slice_imgs[i] for i in sorted(regions_tmp.get("mid", []))]
            bottom_images = [first_phase_slice_imgs[i] for i in sorted(regions_tmp.get("bottom", []))]

            # Atnaujinam split_info.json (deterministiškai)
            corrected_regions = {k: sorted([saved_numbers[i] for i in v]) for k, v in regions_tmp.items()}
            patient_dir = os.path.join(prepared_root_path, patient)
            os.makedirs(patient_dir, exist_ok=True)
            atomic_write_json(os.path.join(patient_dir, "split_info.json"),
                              {"num_slices": len(saved_numbers), **corrected_regions})

            print(f"[{patient}] naudojamas cache (be perrašymo): {len(saved_numbers)} skiltelių")
            return top_images, mid_images, bottom_images

    # 4) Jei čia – arba force_recompute=True, arba cache nepilnas: rašome iš naujo tik ko reikia
    top_images: List[np.ndarray] = []
    mid_images: List[np.ndarray] = []
    bottom_images: List[np.ndarray] = []
    first_phase_slice_imgs: List[np.ndarray] = []

    for phase_no, grid in enumerate(grids, start=start_phase):
        phase_dir = os.path.join(prepared_root_path, patient, f"P{phase_no}")
        ensure_dir(phase_dir)

        slice_imgs_this_phase: List[np.ndarray] = []
        for idx in valid_orig_idx:
            box = ref_boxes[idx]
            slc = crop_to_192(grid, box)
            s_num = idx + 1
            out_png = os.path.join(phase_dir, f"S{s_num}.png")
            out_npy = os.path.join(phase_dir, f"S{s_num}.npy")

            # Rašome tik jei reikia (force_recompute=True) ARBA failų nėra
            if force_recompute or (not (os.path.exists(out_png) or os.path.exists(out_npy))):
                save_slice_files(phase_dir, s_num, slc)

            slice_imgs_this_phase.append(slc)

        if phase_no == start_phase:
            first_phase_slice_imgs = slice_imgs_this_phase

    # 5) Regionų skirstymas (pirmos fazės)
    regions_tmp = split_regions(first_phase_slice_imgs)
    corrected_regions = {
        region: sorted([saved_numbers[i] for i in indices])
        for region, indices in regions_tmp.items()
    }

    region_imgs = {
        key: [first_phase_slice_imgs[i] for i in sorted(indices)]
        for key, indices in regions_tmp.items()
    }
    top_images.extend(region_imgs.get("top", []))
    mid_images.extend(region_imgs.get("mid", []))
    bottom_images.extend(region_imgs.get("bottom", []))

    # 6) split_info.json – atominiu būdu
    patient_dir = os.path.join(prepared_root_path, patient)
    os.makedirs(patient_dir, exist_ok=True)
    atomic_write_json(os.path.join(patient_dir, "split_info.json"),
                      {"num_slices": len(saved_numbers), **corrected_regions})

    print(f"[{patient}] eksportuotos skiltelės: {', '.join('S'+str(i) for i in sorted(saved_numbers))}")
    return top_images, mid_images, bottom_images
