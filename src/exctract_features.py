from typing import List, Tuple
from preprocess.preprocess import (
    get_patient_list,
    circles_root_path,
    process_images_by_patient,
)
from preprocess.visualization import Config, plot_metric_distributions
from preprocess.patients import Patients

from feature_extraction.glcm import calculate_glcm_for_multiple_images
from feature_extraction.lbp import extract_lbp_features_from_images
from feature_extraction.glrlm import calculate_glrlm_for_multiple_images
from feature_extraction.fractal import calculate_fractal_features
from feature_extraction.fourier_transform import calculate_fourier_features
from feature_extraction.wavelet import calculate_wavelet_features
from feature_extraction.hog import calculate_hog_features_from_images

import os

def process_single_phase_combination(phase_tuple: Tuple[int, int]) -> None:
    start_phase, end_phase = phase_tuple
    print(f"Processing phases {start_phase} to {end_phase}")

    phase_suffix = f"_phases_{start_phase}_to_{end_phase-1}"

    patients = Patients()
    # Jei nori švariai nuo nulio – išjunk šį load
    patients.load(phase_suffix)

    # DET. pacientų sąrašas
    for patient in get_patient_list(circles_root_path):
        # NAUDOJAM CACHE -> force_recompute=False (greita ir deterministiška)
        global_images, top_images, mid_images, bottom_images = process_images_by_patient(
            patient, start_phase, end_phase, force_recompute=False
        )

        if not global_images:
            print(f"Warning: No images found for patient {patient} phases {start_phase}-{end_phase-1}")
            continue

        feature_extractors = {
            "GLCM": calculate_glcm_for_multiple_images,
            "GLRLM": calculate_glrlm_for_multiple_images,
            "LBP": extract_lbp_features_from_images,
            "Fractal": calculate_fractal_features,
            "Fourier": calculate_fourier_features,
            "Wavelet": calculate_wavelet_features,
            "HOG": calculate_hog_features_from_images,
        }

        image_segments = {
            "global": global_images,
            "top": top_images,
            "mid": mid_images,
            "bottom": bottom_images,
        }

        feature_providers = {}
        for extractor_name, extractor_func in feature_extractors.items():
            feature_providers[extractor_name] = {}
            for segment_name, segment_images in image_segments.items():
                feature_providers[extractor_name][segment_name] = (
                    lambda images=segment_images, func=extractor_func: func(images)
                )

        for provider_name, feature_provider in feature_providers.items():
            for segment_name, feature_fn in feature_provider.items():
                features_dict = feature_fn()
                for feature_name, feature_value in features_dict.items():
                    patients.add_patient_feature(
                        patient,
                        f"{provider_name}_{feature_name}_{segment_name}",
                        feature_value,
                    )

    # Išsaugom rezultatus
    patients.write(suffix=phase_suffix)

    # Paruošiam vizualizacijai (AS/CO klasės)
    configs: List[Config] = [
        {
            "name": feature,
            "values": {
                "AS": patients.get_feture_for_class("AS", feature),
                "CO": patients.get_feture_for_class("CO", feature),
            },
        }
        for feature in patients.get_all_features()
    ]

    # Užtikrinam, kad kelias egzistuoja (viduje funkcija dar kartą užtvirtins)
    os.makedirs("data/visualizations", exist_ok=True)
    plot_metric_distributions(configs, f"data/visualizations/distributions{phase_suffix}.png")


if __name__ == "__main__":
    process_single_phase_combination((1, 26))
    for i in range(1, 26):
        process_single_phase_combination((i, i + 1))

    # Run all phase combinations
    #process_phase_combinations()
