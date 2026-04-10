from typing import Union, Dict, List
import numpy as np
import SimpleITK as sitk
from radiomics import glrlm
from cv2.typing import MatLike
from numpy.typing import NDArray


def reduce_gray_levels(image: MatLike, levels: int) -> MatLike:
    max_value = 255
    factor = max_value / levels
    image = np.ceil(image / factor).astype(int)  # type: ignore
    return image


def calculate_glrlm_for_multiple_images(gray_images: List[MatLike]) -> Dict[str, float]:
    combined_glrlm_matrix = None

    for image in gray_images:
        # Reduce gray levels and create mask
        # image = reduce_gray_levels(image, 32)
        mask = np.ones_like(image, dtype=bool)
        mask[image == 0] = 0

        # Convert numpy arrays to SimpleITK images
        image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

        # Calculate GLRLM for the current 2D image
        glrlm_features = glrlm.RadiomicsGLRLM(
            image_sitk, mask_sitk, **{"force2D": True, "force2Ddimension": 0}
        )
        glrlm_features.settings.update(
            {"binWidth": 1, "force2D": True, "force2Ddimension": 0, "angles": [0]}
        )

        current_glrlm_matrix = glrlm_features._calculateMatrix()
        current_glrlm_matrix = np.sum(current_glrlm_matrix, axis=(0, 1))

        if combined_glrlm_matrix is None:
            combined_glrlm_matrix = current_glrlm_matrix
        else:
            combined_glrlm_matrix += current_glrlm_matrix

    # Use the last image and mask for feature extraction
    features_extractor = glrlm.RadiomicsGLRLM(image_sitk, mask_sitk)
    features_extractor.P_glrlm = combined_glrlm_matrix
    features_extractor._initCalculation()

    return features_extractor.execute()
