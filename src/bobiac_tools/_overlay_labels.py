from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _is_categorical(series: pd.Series) -> bool:
    """Return `True` when a measurement column should be treated as categorical.

    A column is categorical if it has a pandas categorical dtype or is otherwise
    non-numeric (e.g. strings/objects); numeric columns are treated as continuous.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    return not pd.api.types.is_numeric_dtype(series)


def _categorical_palette(cmap: str, n_categories: int) -> list:
    """Return `n_categories` colors sampled from `cmap`.

    Discrete (listed) colormaps such as `tab10` are sliced directly; continuous
    colormaps are sampled evenly across their range.
    """
    colormap = plt.get_cmap(cmap)
    if isinstance(colormap, ListedColormap):
        return list(cast("list", colormap.colors)[:n_categories])
    if n_categories == 1:
        return [colormap(0.0)]
    return [colormap(i / (n_categories - 1)) for i in range(n_categories)]


def _stack_channels(channels: list[np.ndarray]) -> np.ndarray:
    """Normalize up to 3 single-channel 2-D arrays and stack into an (H, W, 3) image.

    Each channel is normalized independently to [0, 1] and placed into the R, G,
    B planes in order; absent planes are filled with zeros.

    Raises
    ------
    ValueError
        If more than 3 channels are supplied.
    """
    if len(channels) > 3:
        raise ValueError(
            f"At most 3 channels are supported when passing a list, got {len(channels)}"
        )
    h, w = channels[0].shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for i, ch in enumerate(channels):
        ch = ch.astype(float)
        rgb[..., i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return rgb


def overlay_labels(
    image: np.ndarray | list[np.ndarray],
    label_mask: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    id_col: str | None = None,
    measurement_col: str | None = None,
    alpha: float = 0.3,
    cmap: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Overlay segmentation labels on an image, color-coded by a measurement.

    Parameters
    ----------
    image : np.ndarray or list of np.ndarray
        Grayscale `(H, W)`, RGB `(H, W, 3)` image, or a list of up to 3
        grayscale `(H, W)` arrays treated as individual color channels (R, G,
        B in order). Each channel is normalized independently.
    label_mask : np.ndarray or None
        Integer array where each cell has a unique ID; background is 0. If None,
        only the image is shown.
    df : pd.DataFrame or None
        Table with cell IDs and measurements. If None, cells are shown with
        random colors and labeled by ID (segmentation validation mode).
    id_col : str or None
        Column in `df` containing cell IDs matching values in `label_mask`.
    measurement_col : str or None
        Column in `df` to color cells by.
    alpha : float
        Opacity of the colored overlay (0 = transparent, 1 = opaque). By default, 0.3.
    cmap : str or None
        Matplotlib colormap. Defaults: categorical `"tab10"`, continuous
        `"viridis"`.
    figsize : tuple of float
        Figure size in inches. By default, `(8, 8)`.
    vmin : float or None
        Minimum value for the continuous colormap range. Defaults to the data minimum.
    vmax : float or None
        Maximum value for the continuous colormap range. Defaults to the data maximum.
    show : bool
        If True, call ``plt.show()`` before returning. By default, True. Set to
        False to control display yourself (e.g. ``fig.savefig(...)`` or automatic
        inline rendering in notebooks).

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes the overlay was drawn on.
    """
    # --- build display image ---
    if isinstance(image, list):
        img_display = _stack_channels(image)
    else:
        img_display = image.astype(float)
        img_display = (img_display - img_display.min()) / (
            img_display.max() - img_display.min() + 1e-8
        )
        if img_display.ndim == 2:
            img_display = np.stack([img_display] * 3, axis=-1)

    # --- no overlay: just show the image ---
    if label_mask is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_display)
        ax.axis("off")
        if show:
            plt.show()
        return fig, ax

    overlay = np.zeros((*label_mask.shape, 4), dtype=float)  # RGBA

    # detect the coloring mode once and reuse it for the overlay and annotations
    is_categorical = df is not None and _is_categorical(df[measurement_col])
    centroids: dict = {}
    categories: np.ndarray = np.empty(0)
    color_map: dict = {}
    norm: mcolors.Normalize | None = None

    # --- random colors: segmentation validation mode ---
    if df is None:
        rng = np.random.default_rng(42)  # fixed seed for reproducible colors
        cell_ids = np.unique(label_mask[label_mask > 0])
        colors = rng.random((len(cell_ids), 3))
        for cell_id, color in zip(cell_ids, colors):
            overlay[label_mask == cell_id] = [*color, alpha]
            centroids[cell_id] = np.argwhere(label_mask == cell_id).mean(axis=0)

    # --- categorical measurement ---
    elif is_categorical:
        cmap = cmap or "tab10"
        categories = df[measurement_col].unique()
        color_map = dict(zip(categories, _categorical_palette(cmap, len(categories))))
        for cell_id, val in zip(df[id_col], df[measurement_col]):
            overlay[label_mask == cell_id] = [*mcolors.to_rgb(color_map[val]), alpha]

    # --- continuous measurement ---
    else:
        cmap = cmap or "viridis"
        values = df[measurement_col]
        norm = mcolors.Normalize(
            vmin=vmin if vmin is not None else values.min(),
            vmax=vmax if vmax is not None else values.max(),
        )
        colormap = plt.get_cmap(cmap)
        for cell_id, value in zip(df[id_col], values):
            overlay[label_mask == cell_id] = [*colormap(norm(value))[:3], alpha]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_display)
    ax.imshow(overlay)
    ax.axis("off")

    if df is None:
        for cell_id, yx in centroids.items():
            ax.text(
                yx[1],
                yx[0],
                str(cell_id),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
        ax.set_title("Segmentation overlay")
    elif is_categorical:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[cat],
                label=cat,
                markersize=10,
            )
            for cat in categories
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(str(measurement_col))
    else:
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=measurement_col,
            shrink=0.7,
        )
        ax.set_title(str(measurement_col))

    if show:
        plt.show()
    return fig, ax
