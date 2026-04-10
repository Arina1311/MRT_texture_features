import numpy as np
import pywt


def calculate_combined_mean_sd(accum_sum, accum_sum_squares, total_elements):
    """
    Calculate the combined mean and standard deviation from accumulated sums.

    Parameters:
    - accum_sum: Accumulated sum of pixel values across the image set.
    - accum_sum_squares: Accumulated sum of squared pixel values across the image set.
    - total_elements: Total number of elements (pixels) across all images in the set.

    Returns:
    - mean: Combined mean across the image set.
    - sd: Combined standard deviation across the image set.
    """
    mean = accum_sum / total_elements
    variance = (accum_sum_squares / total_elements) - mean**2
    sd = np.sqrt(variance)
    return mean, sd


def calculate_wavelet_features(image_list, wavelet="haar", level=2):
    # Initialize accumulators for each level
    accumulators = {
        f"level_{i}": {
            "LL_sum": 0,
            "LL_sum_squares": 0,
            "LL_total_elements": 0,
            "LH_sum": 0,
            "LH_sum_squares": 0,
            "LH_total_elements": 0,
            "HL_sum": 0,
            "HL_sum_squares": 0,
            "HL_total_elements": 0,
            "HH_sum": 0,
            "HH_sum_squares": 0,
            "HH_total_elements": 0,
        }
        for i in range(1, level + 1)
    }

    for image_array in image_list:
        coeffs = pywt.wavedec2(image_array, wavelet, level=level)
        for i, coeff in enumerate(coeffs):
            if i == 0:
                LL = coeff
                accumulators[f"level_{level}"]["LL_sum"] += np.sum(LL)
                accumulators[f"level_{level}"]["LL_sum_squares"] += np.sum(LL**2)
                accumulators[f"level_{level}"]["LL_total_elements"] += LL.size
            else:
                LH, HL, HH = coeff
                current_level = level - i + 1
                for band, name in zip([LH, HL, HH], ["LH", "HL", "HH"]):
                    accumulators[f"level_{current_level}"][f"{name}_sum"] += np.sum(
                        band
                    )
                    accumulators[f"level_{current_level}"][
                        f"{name}_sum_squares"
                    ] += np.sum(band**2)
                    accumulators[f"level_{current_level}"][
                        f"{name}_total_elements"
                    ] += band.size

    # Compute combined mean and standard deviation for each sub-band at each level
    combined_features = {}
    for level_num in range(1, level + 1):
        level_key = f"level_{level_num}"
        for band in ["LL", "LH", "HL", "HH"]:
            if band == "LL" and level_num != level:
                continue
            mean, sd = calculate_combined_mean_sd(
                accumulators[level_key][f"{band}_sum"],
                accumulators[level_key][f"{band}_sum_squares"],
                accumulators[level_key][f"{band}_total_elements"],
            )
            combined_features[f"{level_key}_{band}_mean"] = mean
            combined_features[f"{level_key}_{band}_sd"] = sd

    return combined_features
