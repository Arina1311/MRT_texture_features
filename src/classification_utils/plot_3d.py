import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from class_mapping import map_numeric_to_classname


def plot_misclassifications_3d(
    X_3d,
    y_true,
    y_pred,
    sample_ids,
    confidences,
    method_name,
    output_dir="plots",
):
    os.makedirs(output_dir, exist_ok=True)

    # Build a DataFrame for easy grouping
    df_plot = pd.DataFrame(
        {
            "UMAP1": X_3d[:, 0],
            "UMAP2": X_3d[:, 1],
            "UMAP3": X_3d[:, 2],
            "true_label": [map_numeric_to_classname(lbl) for lbl in y_true],
            "pred_label": [map_numeric_to_classname(lbl) for lbl in y_pred],
            "confidence": confidences,
            "sample_id": sample_ids,
        }
    )

    df_plot["correct"] = df_plot["true_label"] == df_plot["pred_label"]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    # Plot correct classifications
    correct_df = df_plot[df_plot["correct"]]
    for label in sorted(correct_df["true_label"].unique()):
        subset = correct_df[correct_df["true_label"] == label]
        color = "blue" if label == "AS" else "green"

        hover = (
            subset["sample_id"]
            + "<br>Confidence: "
            + (subset["confidence"] * 100).round(1).astype(str)
            + "%"
        )

        fig.add_trace(
            go.Scatter3d(
                x=subset["UMAP1"],
                y=subset["UMAP2"],
                z=subset["UMAP3"],
                mode="markers",
                marker=dict(size=7, color=color, line=dict(color="black", width=1)),
                name=f"Correct {label}",
                text=hover,
                hoverinfo="text+name",
            )
        )

    # Plot misclassified samples
    incorrect_df = df_plot[~df_plot["correct"]]
    if not incorrect_df.empty:
        for true_lab in sorted(incorrect_df["true_label"].unique()):
            subset = incorrect_df[incorrect_df["true_label"] == true_lab]
            color = "blue" if true_lab == "AS" else "green"

            hover = (
                subset["sample_id"]
                + "<br>True: "
                + subset["true_label"]
                + "<br>Pred: "
                + subset["pred_label"]
                + "<br>Confidence: "
                + (subset["confidence"] * 100).round(1).astype(str)
                + "%"
            )

            fig.add_trace(
                go.Scatter3d(
                    x=subset["UMAP1"],
                    y=subset["UMAP2"],
                    z=subset["UMAP3"],
                    mode="markers",
                    marker=dict(
                        size=11,
                        symbol="x",
                        color=color,
                        line=dict(color="red", width=2),
                    ),
                    name=f"Misclassified {true_lab}",
                    text=hover,
                    hoverinfo="text+name",
                )
            )

    fig.update_layout(
        title=f"UMAP 3D Visualization - {method_name}",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )

    html_path = os.path.join(output_dir, f"{method_name}_misclassifications_3d.html")
    fig.write_html(html_path)

    png_path = os.path.join(output_dir, f"{method_name}_misclassifications_3d.png")
    fig.write_image(png_path)

    return fig
