from typing import Literal, Union
import numpy as np
from skimage import feature
import plotly.graph_objects as go
from cv2.typing import MatLike
from numpy.typing import NDArray

GLCM = NDArray[Union[np.uint32, np.float64]]


def reduce_gray_levels(image: MatLike, levels: int) -> MatLike:
    max_value = 255
    factor = max_value / levels
    reduced_image = np.ceil(image / factor).astype(int)  # type: ignore
    return reduced_image


def calculate_glcm(
    gray_image: MatLike, distance=1, angle=0, number_of_gray_levels=32
) -> GLCM:
    quantized_image = reduce_gray_levels(gray_image, number_of_gray_levels)
    glcm = feature.graycomatrix(
        quantized_image,
        distances=[distance],
        angles=[angle],
        symmetric=True,
        normed=False,
        levels=number_of_gray_levels + 1,
    )
    return glcm[
        1:, 1:
    ]  # Drop first row and first column since 0 level pixels should be ignored


def calculate_glcm_for_multiple_images(
    gray_images: list[MatLike], distance=1, angle=0, number_of_gray_levels=32
) -> dict[str, float]:
    combined_glcm = np.zeros(
        (number_of_gray_levels + 1, number_of_gray_levels + 1, 1, 1), dtype=np.uint32
    )

    for gray_image in gray_images:
        quantized_image = reduce_gray_levels(gray_image, number_of_gray_levels)
        glcm = feature.graycomatrix(
            quantized_image,
            distances=[distance],
            angles=[angle],
            symmetric=True,
            normed=False,
            levels=number_of_gray_levels + 1,
        )
        combined_glcm += glcm

    # Drop first row and first column since 0 level pixels should be ignored
    adjusted_glcm = combined_glcm[1:, 1:]

    return {
        "dissimilarity": get_property_from_glcm(adjusted_glcm, "dissimilarity"),
        "energy": get_property_from_glcm(adjusted_glcm, "energy"),
        "homogeneity": get_property_from_glcm(adjusted_glcm, "homogeneity"),
        "contrast": get_property_from_glcm(adjusted_glcm, "contrast"),
        "correlation": get_property_from_glcm(adjusted_glcm, "correlation"),
        "ASM": get_property_from_glcm(adjusted_glcm, "ASM"),
    }


GlcmFeatures = Literal[
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"
]


def get_property_from_glcm(glcm: GLCM, property_name: GlcmFeatures) -> float:
    return feature.graycoprops(glcm, property_name)[0][0]


def plot_glcm(glcm: GLCM) -> None:
    glcm_matrix = glcm[:, :, 0, 0]

    # Create row and column headers (gray levels)
    gray_levels = [str(i) for i in range(31)]

    # Create a table with Plotly
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Gray Level i \\ j"] + gray_levels),
                cells=dict(
                    values=[gray_levels] + [glcm_matrix[:, i] for i in range(31)]
                ),
            )
        ]
    )

    fig.update_layout(
        title="Gray-Level Co-occurrence Matrix (GLCM)",
        width=900,  # Adjust the width and height to your preference
        height=900,
    )

    fig.show()
