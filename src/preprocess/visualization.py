from typing import TypedDict, Dict, List
import os
import matplotlib.pyplot as plt

class Config(TypedDict):
    name: str
    values: Dict[str, List[float]]  # pvz. {"AS": [...], "CO": [...]}

def plot_metric_distributions(configs: List[Config], output_path: str) -> None:
    # Paprastas pavyzdinis braižymas — pritaikyk pagal savo realesnę vizualizaciją
    plt.figure()
    for cfg in configs:
        vals_as = cfg["values"].get("AS", [])
        vals_co = cfg["values"].get("CO", [])
        # pvz., dėmėtas boxplot, arba vidurkiai — čia tik placeholderis
        plt.scatter([cfg["name"]]*len(vals_as), vals_as, marker="o")
        plt.scatter([cfg["name"]]*len(vals_co), vals_co, marker="x")

    # Užtikriname, kad aplankas yra
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
