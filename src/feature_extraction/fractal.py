import numpy as np
from skimage.util import view_as_blocks


def fractal_dimension(image: np.ndarray) -> np.float64:
    # Ensure the image is grayscale and of type float
    image = image.astype(float)

    sizes = np.array([2, 4, 8, 16, 32, 64])
    counts = []

    for size in sizes:
        # Pad the image to ensure it's divisible by size
        h, w = image.shape
        pad_h = (size - h % size) % size
        pad_w = (size - w % size) % size
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")

        # View the image as blocks
        blocks = view_as_blocks(padded, block_shape=(size, size))

        # Count the number of boxes needed
        count = np.sum(np.any(blocks > image.mean(), axis=(2, 3)))
        counts.append(count)

    # Fit a line to log-log plot
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]


def calculate_fractal_features(images):
    dimensions = []
    for image in images:
        dimension = fractal_dimension(image)
        dimensions.append(dimension)

    stats = {
        "mean": np.mean(dimensions),
        "variance": np.var(dimensions),
    }

    return stats
