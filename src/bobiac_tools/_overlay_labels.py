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


# Additive composite colors for multi-channel image display:
# 1 channel: grey | 2: red + cyan | 3: magenta + cyan + yellow
_CHANNEL_COLORS: dict[int, list[tuple[float, float, float]]] = {
    1: [(1.0, 1.0, 1.0)],
    2: [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)],
    3: [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)],
}

# Fixed colors for multi-mask display (up to 5 masks).
_MASK_COLORS: list[tuple[float, float, float]] = [
    (1.00, 0.10, 0.11),  # red
    (0.42, 0.67, 1.00),  # blue
    (0.20, 1.00, 0.17),  # green
    (0.60, 0.31, 0.64),  # purple
    (1.00, 0.50, 0.00),  # orange
]


def _stack_channels(channels: list[np.ndarray]) -> np.ndarray:
    """Composite up to 3 single-channel 2-D arrays into an (H, W, 3) RGB image.

    Each channel is normalized independently to [0, 1], then additively blended
    using semantic colors from ``_CHANNEL_COLORS``:

    - 1 channel : grey
    - 2 channels : red + cyan
    - 3 channels : magenta + cyan + yellow

    Raises
    ------
    ValueError
        If more than 3 channels are supplied.
    """
    n = len(channels)
    if n > 3:
        raise ValueError(
            f"At most 3 channels are supported when passing a list, got {n}"
        )
    colors = _CHANNEL_COLORS[n]
    h, w = channels[0].shape
    composite = np.zeros((h, w, 3), dtype=float)
    for ch, color in zip(channels, colors):
        ch = ch.astype(float)
        normalized = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        for c in range(3):
            composite[..., c] += normalized * color[c]
    return np.clip(composite, 0.0, 1.0)


def _plot_coordinates(
    ax: Axes,
    coords: np.ndarray | pd.DataFrame | list[np.ndarray] | None,
) -> None:
    """Scatter *coords* on *ax*.

    - ``list`` of arrays : one color per entry (tab10), legend with index.
    - ``pd.DataFrame`` with ``'x'``/``'y'`` columns : optional ``'channel'``
      column for per-channel coloring.
    - plain array : (N, 2) in ``(row, col)`` order, plotted in yellow.
    """
    if coords is None:
        return
    cmap = plt.get_cmap("tab10")
    tab10: list = list(cmap(np.linspace(0, 1, 10)))
    if isinstance(coords, list):
        for i, arr in enumerate(coords):
            arr = np.asarray(arr)
            ax.scatter(
                arr[:, 1], arr[:, 0], s=10,
                c=[tab10[i % len(tab10)]],
                linewidths=0.5, edgecolors="black", label=str(i),
            )
        ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left", title="channel")
    elif isinstance(coords, pd.DataFrame):
        if "channel" in coords.columns:
            for i, ch in enumerate(coords["channel"].unique()):
                sub = coords[coords["channel"] == ch]
                ax.scatter(
                    sub["x"], sub["y"], s=10,
                    c=[tab10[i % len(tab10)]],
                    linewidths=0.5, edgecolors="black", label=str(ch),
                )
            ax.legend(bbox_to_anchor=(1.01, 0), loc="lower left", title="channel")
        else:
            ax.scatter(coords["x"], coords["y"], s=10, c="yellow",
                       linewidths=0.5, edgecolors="black")
    else:
        coords = np.asarray(coords)
        ax.scatter(coords[:, 1], coords[:, 0], s=10, c="yellow",
                   linewidths=0.5, edgecolors="black")


def _compute_crop(
    label_mask: np.ndarray | list[np.ndarray],
    focus_object: int,
) -> tuple[int, int, int, int] | None:
    """Return ``(y0, y1, x0, x1)`` bounding box for *focus_object* with 15 % padding.

    Searches all masks in *label_mask* (list or single array). Returns ``None``
    if the object is not found.
    """
    masks = label_mask if isinstance(label_mask, list) else [label_mask]
    for m in masks:
        region = np.argwhere(m == focus_object)
        if len(region) > 0:
            y_min, x_min = region.min(axis=0)
            y_max, x_max = region.max(axis=0)
            pad = int(max(y_max - y_min, x_max - x_min) * 0.15)
            h, w = m.shape
            return (
                max(0, y_min - pad), min(h, y_max + pad),
                max(0, x_min - pad), min(w, x_max + pad),
            )
    return None


def _apply_crop(
    arr: np.ndarray | None,
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    """Slice *arr* to the ``(y0, y1, x0, x1)`` *crop* region."""
    if arr is None or crop is None:
        return arr
    y0, y1, x0, x1 = crop
    return arr[y0:y1, x0:x1]


def _offset_coordinates(
    coords: np.ndarray | pd.DataFrame | list[np.ndarray] | None,
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray | pd.DataFrame | list[np.ndarray] | None:
    """Filter *coords* to the crop region and shift to the cropped coordinate system."""
    if coords is None or crop is None:
        return coords
    y0, y1, x0, x1 = crop
    if isinstance(coords, list):
        result = []
        for arr in coords:
            arr = np.asarray(arr, dtype=float).copy()
            in_crop = (
                (arr[:, 0] >= y0) & (arr[:, 0] < y1) &
                (arr[:, 1] >= x0) & (arr[:, 1] < x1)
            )
            arr = arr[in_crop]
            arr[:, 0] -= y0
            arr[:, 1] -= x0
            result.append(arr)
        return result
    elif isinstance(coords, pd.DataFrame):
        df = coords.copy()
        in_crop = (
            (df["x"] >= x0) & (df["x"] < x1) &
            (df["y"] >= y0) & (df["y"] < y1)
        )
        df = df[in_crop].copy()
        df["x"] -= x0
        df["y"] -= y0
        return df
    else:
        coords = np.asarray(coords, dtype=float).copy()
        in_crop = (
            (coords[:, 0] >= y0) & (coords[:, 0] < y1) &
            (coords[:, 1] >= x0) & (coords[:, 1] < x1)
        )
        coords = coords[in_crop]
        coords[:, 0] -= y0
        coords[:, 1] -= x0
        return coords


def overlay_labels(
    image: np.ndarray | list[np.ndarray] | None = None,
    label_mask: np.ndarray | list[np.ndarray] | None = None,
    df: pd.DataFrame | None = None,
    id_col: str | None = None,
    measurement_col: str | None = None,
    alpha: float = 0.3,
    image_cmap: str | None = None,
    mask_cmap: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    coordinates: np.ndarray | pd.DataFrame | list[np.ndarray] | None = None,
    focus_object: int | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Overlay segmentation labels on an image, color-coded by a measurement.

    Parameters
    ----------
    image : np.ndarray, list of np.ndarray, or None
        One of:

        - 2-D ``(H, W)`` grayscale array.
        - 3-D ``(H, W, 3)`` RGB array.
        - 3-D ``(C, H, W)`` array (C ≤ 3) — automatically split into channels.
        - List of up to 3 grayscale ``(H, W)`` arrays — additively composited
          using semantic colors (1: grey | 2: red + cyan | 3: magenta + cyan + yellow).
        - ``None`` — no background image (a black canvas is used).
    label_mask : np.ndarray, list of np.ndarray, or None
        One of:

        - 2-D integer array where each cell has a unique ID; background is 0.
        - 3-D ``(C, H, W)`` array (C ≤ 3) — split into a list of masks.
        - List of masks — each displayed in a distinct color (up to 5).
        - ``None`` — no overlay, only the image is shown.
    df : pd.DataFrame or None
        Table with cell IDs and measurements. Only used with a single
        *label_mask*. If ``None``, cells are colored randomly and labeled by ID.
    id_col : str or None
        Column in *df* with cell IDs matching *label_mask* values.
    measurement_col : str or None
        Column in *df* to color cells by (categorical or continuous).
    alpha : float
        Overlay opacity (0 = transparent, 1 = opaque). By default, 0.3.
    image_cmap : str or None
        Colormap for the base image. Only applies to single grayscale ``(H, W)``
        input; ignored for RGB arrays and multi-channel lists. Defaults to
        ``"gray"`` for grayscale input.
    mask_cmap : str or None
        Colormap for the label overlay. Defaults: categorical ``"tab10"``,
        continuous ``"viridis"``.
    figsize : tuple of float
        Figure size in inches. Adjusted automatically when *focus_object* is set.
    vmin : float or None
        Min value for the continuous colormap range. Defaults to data minimum.
    vmax : float or None
        Max value for the continuous colormap range. Defaults to data maximum.
    coordinates : array, pd.DataFrame, list of arrays, or None
        Points to scatter on top of the image.

        - Plain ``(N, 2)`` array or list of arrays: ``(row, col)`` order.
        - ``pd.DataFrame`` with ``'x'`` / ``'y'`` columns; optional ``'channel'``
          column for per-channel coloring.
    focus_object : int or None
        Object ID to zoom into. Crops all data to the object's bounding box
        (with 15 % padding) and resizes the figure to match.
    show : bool
        If ``True``, call ``plt.show()`` before returning. By default, ``True``.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes the overlay was drawn on.

    Raises
    ------
    ValueError
        If both *image* and *label_mask* are ``None``.
    """
    if image is None and label_mask is None:
        raise ValueError("At least one of 'image' or 'label_mask' must be provided.")

    # --- normalize 3D (C, H, W) arrays to lists of 2-D slices ---
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] <= 3:
        image = [image[i] for i in range(image.shape[0])]
    if (
        isinstance(label_mask, np.ndarray)
        and label_mask.ndim == 3
        and label_mask.shape[0] <= 3
    ):
        label_mask = [label_mask[i] for i in range(label_mask.shape[0])]

    # --- build display image ---
    if isinstance(image, list):
        img_display = _stack_channels(image)
        _image_cmap = None  # composite is already RGB
    elif image is not None:
        img_display = image.astype(float)
        img_display = (img_display - img_display.min()) / (
            img_display.max() - img_display.min() + 1e-8
        )
        _image_cmap = (image_cmap or "gray") if img_display.ndim == 2 else None
    else:
        img_display = None
        _image_cmap = None

    # --- compute crop region from focus_object ---
    crop: tuple[int, int, int, int] | None = None
    if focus_object is not None and label_mask is not None:
        crop = _compute_crop(label_mask, focus_object)

    img_display = _apply_crop(img_display, crop)
    coordinates = _offset_coordinates(coordinates, crop)

    _figsize = figsize
    if crop is not None:
        y0, y1, x0, x1 = crop
        w_px, h_px = x1 - x0, y1 - y0
        scale = max(figsize)
        _figsize = (scale * w_px / max(w_px, h_px), scale * h_px / max(w_px, h_px))

    # --- multiple masks: one fixed color per mask ---
    if isinstance(label_mask, list):
        cropped_masks = [
            cast("np.ndarray", _apply_crop(m, crop)) for m in label_mask
        ]
        h, w = cropped_masks[0].shape
        mask_overlay = np.zeros((h, w, 4))
        for m, color in zip(cropped_masks, _MASK_COLORS):
            region = m > 0
            mask_overlay[region, :3] = color
            mask_overlay[region, 3] = alpha
        _img = img_display if img_display is not None else np.zeros((h, w, 3))
        fig, ax = plt.subplots(figsize=_figsize)
        ax.imshow(_img, cmap=_image_cmap)
        ax.imshow(mask_overlay)
        ax.axis("off")
        _plot_coordinates(ax, coordinates)
        if show:
            plt.show()
        return fig, ax

    # --- no mask: show image only ---
    if label_mask is None:
        fig, ax = plt.subplots(figsize=_figsize)
        ax.imshow(img_display, cmap=_image_cmap)  # type: ignore[arg-type]
        ax.axis("off")
        _plot_coordinates(ax, coordinates)
        if show:
            plt.show()
        return fig, ax

    # --- single mask + optional measurement overlay ---
    label_mask = cast("np.ndarray", _apply_crop(label_mask, crop))
    h, w = label_mask.shape
    _img = img_display if img_display is not None else np.zeros((h, w, 3))
    mask_overlay = np.zeros((h, w, 4), dtype=float)

    is_categorical = df is not None and _is_categorical(df[measurement_col])
    centroids: dict = {}
    categories: np.ndarray = np.empty(0)
    color_map: dict = {}
    norm: mcolors.Normalize | None = None

    # --- random colors: segmentation validation mode ---
    if df is None:
        rng = np.random.default_rng(42)
        cell_ids = np.unique(label_mask[label_mask > 0])
        colors = rng.random((len(cell_ids), 3))
        for cell_id, color in zip(cell_ids, colors):
            mask_overlay[label_mask == cell_id] = [*color, alpha]
            centroids[cell_id] = np.argwhere(label_mask == cell_id).mean(axis=0)

    # --- categorical measurement ---
    elif is_categorical:
        mask_cmap = mask_cmap or "tab10"
        categories = df[measurement_col].unique()
        palette = _categorical_palette(mask_cmap, len(categories))
        color_map = dict(zip(categories, palette))
        for cell_id, val in zip(df[id_col], df[measurement_col]):
            rgba = [*mcolors.to_rgb(color_map[val]), alpha]
            mask_overlay[label_mask == cell_id] = rgba

    # --- continuous measurement ---
    else:
        mask_cmap = mask_cmap or "viridis"
        values = df[measurement_col]
        norm = mcolors.Normalize(
            vmin=vmin if vmin is not None else values.min(),
            vmax=vmax if vmax is not None else values.max(),
        )
        colormap = plt.get_cmap(mask_cmap)
        for cell_id, value in zip(df[id_col], values):
            mask_overlay[label_mask == cell_id] = [*colormap(norm(value))[:3], alpha]

    # --- plot ---
    fig, ax = plt.subplots(figsize=_figsize)
    ax.imshow(_img, cmap=_image_cmap)
    ax.imshow(mask_overlay)
    ax.axis("off")

    if df is None:
        for cell_id, yx in centroids.items():
            ax.text(
                yx[1], yx[0], str(cell_id),
                ha="center", va="center", fontsize=6, color="white",
            )
        ax.set_title("Segmentation overlay")
    elif is_categorical:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[cat], label=cat, markersize=10)
            for cat in categories
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(str(measurement_col))
    else:
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=mask_cmap),
            ax=ax, label=measurement_col, shrink=0.7,
        )
        ax.set_title(str(measurement_col))

    _plot_coordinates(ax, coordinates)
    if show:
        plt.show()
    return fig, ax
