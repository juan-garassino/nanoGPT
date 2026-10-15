"""Weight-matrix heatmap plotting for GPT state dicts."""

import os
import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch


def plot_heatmaps(
    data: Dict[str, torch.Tensor],
    output_folder: str,
    num_heads: int = 4,
    title_font_size: int = 12,
    subtitle_font_size: int = 10,
    figsize: Tuple[int, int] = (120, 40),
    dpi: int = 100,
    row_spacing: float = 0.2,
    column_spacing: float = 0.2,
    title_spacing: float = 0.4,
    subtitle_spacing: float = 0.3,
    tick_font_size: int = 8,
) -> None:
    """Save a grid of heatmaps of a model's ``*.weight`` tensors as a timestamped PNG.

    The grid has ``len(weight keys) // 4`` rows and ``num_heads`` columns, and the
    figure is written to
    ``$HOME/Code/juan-garassino/mySandbox/nanoGPT/plots/<output_folder>/heatmaps[<timestamp>].png``.
    Panels are indexed ``row = i % rows``, ``col = i // rows``, so the number of
    weight tensors must not exceed ``rows * num_heads``.

    Parameters
    ----------
    data : dict of {str: torch.Tensor}
        State dict (e.g. ``model.state_dict()``); each ``*.weight`` tensor is
        flattened to 2-D with shape ``(shape[0], -1)``.
    output_folder : str
        Sub-folder of ``plots/`` to write into; created if missing.
    num_heads : int, default 4
        Number of columns in the grid.
    title_font_size : int
        Unused.
    subtitle_font_size : int
        Font size of each panel title.
    figsize : tuple of int
        Figure size in inches ``(width, height)``.
    dpi : int
        Resolution of the figure and the saved PNG.
    row_spacing, column_spacing : float
        ``hspace`` / ``wspace`` passed to ``subplots_adjust``.
    title_spacing : float
        Figure top margin is ``1 - title_spacing / figsize[1]``.
    subtitle_spacing : float
        Padding of each panel title.
    tick_font_size : int
        Font size of axis ticks.
    """
    output_path = os.path.join(
        os.environ.get("HOME"),
        "Code",
        "juan-garassino",
        "mySandbox",
        "nanoGPT",
        "plots",
        output_folder,
    )

    os.makedirs(output_path, exist_ok=True)

    for number, key in enumerate(data.keys()):
        print(number + 1, key)

    # Retrieve the keys of the linear layers
    linear_layer_keys = [key for key in data.keys() if key.endswith(".weight")]

    # Calculate the number of linear layers and heads

    num_layers = len(linear_layer_keys) // 4

    print(num_layers)

    print(num_heads)

    # Create the subplot grid
    fig, axes = plt.subplots(num_layers, num_heads, figsize=figsize, dpi=dpi)

    # Adjust the row and column spacing
    fig.subplots_adjust(
        hspace=row_spacing, wspace=column_spacing, top=1 - title_spacing / figsize[1]
    )

    # Iterate over the linear layers and plot the heatmaps
    for i, layer_key in enumerate(linear_layer_keys):
        weights = data[layer_key]

        weights = weights.cpu()

        # flatten all but the first dimension so 1-D tensors become (n, 1)
        weights_2d = weights.view(weights.size(0), -1)

        weights_np = weights_2d.detach().numpy()

        # fill the grid column by column
        layer_idx = i % num_layers
        head_idx = i // num_layers

        axes[layer_idx, head_idx].imshow(weights_np, cmap="hot")
        axes[layer_idx, head_idx].set_title(
            f"Layer: {layer_idx}\nHead: {head_idx+1}",
            fontsize=subtitle_font_size,
            pad=subtitle_spacing,
        )

        axes[layer_idx, head_idx].tick_params(
            axis="both", which="both", labelsize=tick_font_size
        )

    fig.tight_layout()

    # Save the plot with timestamp in the output folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(output_path, f"heatmaps[{timestamp}].png")
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved heatmaps at: {output_path}")
