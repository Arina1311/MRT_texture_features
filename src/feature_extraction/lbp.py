from skimage import feature
import numpy as np


def extract_lbp_features(image, radius, n_points, method="uniform"):
    lbp = feature.local_binary_pattern(image, n_points, radius, method)
    n_bins = int(lbp.max() + 1)  # Number of unique patterns
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def get_combined_lbp(images):
    summed_histogram = None
    num_images = len(images)
    for image in images:
        hist = extract_lbp_features(image, 1, 8)
        if summed_histogram is None:
            summed_histogram = hist
        else:
            summed_histogram += hist

    # Normalize the summed histogram
    if num_images > 0:
        normalized_histogram = summed_histogram / num_images
    else:
        normalized_histogram = summed_histogram  # or np.zeros_like(summed_histogram)

    return normalized_histogram


def extract_lbp_features_from_images(images):
    combined_histogram = get_combined_lbp(images)

    uniformity = np.sum(combined_histogram**2)
    contrast = np.sum(
        [
            ((i - j) ** 2) * combined_histogram[i] * combined_histogram[j]
            for i in range(len(combined_histogram))
            for j in range(len(combined_histogram))
        ]
    )

    return {"uniformity": uniformity, "contrast": contrast}
