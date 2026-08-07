import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

def plot_token_heatmap(
    words,
    importances,
    label=None,
    output_dir="explanations",
    filename="heatmap.pdf",
    label_names=None,
    predicted_label=None
):
    os.makedirs(output_dir, exist_ok=True)

    gt_str = label_names.get(label, str(label)) if label_names and label is not None else str(label)
    pred_str = label_names.get(predicted_label, str(predicted_label)) if label_names and predicted_label is not None else ""

    importance_matrix = np.array([importances])

    fig = plt.figure(figsize=(max(12, len(words) * 0.6), 0.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.3, len(words)], wspace=0.01)


    ax_label = fig.add_subplot(gs[0])
    ax_label.axis('off')
    ax_label.text(
        1.0, 0.5,
        f"Predicted:\n{pred_str}",
        fontsize=9,
        ha='right',
        va='center'
    )


    ax = fig.add_subplot(gs[1])
    sns.heatmap(
        importance_matrix,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",    #ff
        cbar=False,
        xticklabels=words,
        yticklabels=[],
        ax=ax
    )

    ax.set_title(f"Ground Truth: {gt_str}", fontsize=9)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved token heatmap to: {save_path}")

