import numpy as np


def calculate_fourier_features(image_list):
    combined_fft = None

    for image_array in image_list:
        if image_array.ndim != 2:
            raise ValueError("Each image must be a 2D grayscale image.")

        # Apply the Fourier Transform to the current image
        fft_image = np.fft.fft2(image_array)
        fft_image_shifted = np.fft.fftshift(fft_image)

        # Accumulate the Fourier Transform
        if combined_fft is None:
            combined_fft = fft_image_shifted
        else:
            combined_fft += fft_image_shifted

    combined_fft /= len(image_list)

    return _calculate_features_from_fft(combined_fft)


def _calculate_features_from_fft(fft_matrix):
    magnitude_spectrum = np.abs(fft_matrix)

    M, N = magnitude_spectrum.shape
    total_elements = M * N

    # Calculate Mean
    mean = np.sum(magnitude_spectrum) / total_elements

    # Calculate Variance
    variance = np.sum((magnitude_spectrum - mean) ** 2) / total_elements

    # Calculate Energy
    energy = np.sum(magnitude_spectrum**2)

    # Calculate Entropy
    p_uv = magnitude_spectrum / np.sum(
        magnitude_spectrum
    )  # Normalized magnitude spectrum
    p_uv = p_uv[p_uv > 0]  # Avoid log(0) by considering only non-zero probabilities
    entropy = -np.sum(p_uv * np.log(p_uv))

    # Return all features in a dictionary
    features = {
        "Mean": mean,
        "Variance": variance,
        "Energy": energy,
        "Entropy": entropy,
    }

    return features
