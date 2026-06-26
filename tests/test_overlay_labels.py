"""Tests for the ``overlay_labels`` plotting helper.

The overlay modes are exercised on the real ``F01_202w1`` image/label pair that
lives next to this file; small synthetic arrays are used for the checks that need
exact, controlled pixel values.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bobiac_tools import overlay_labels
from bobiac_tools._overlay_labels import _categorical_palette, _is_categorical

# overlay_labels shows by default; Agg has no GUI, so render off-screen.
plt.switch_backend("Agg")

DATA_DIR = Path(__file__).parent
IMAGE_PATH = DATA_DIR / "F01_202w1_image.TIF"
LABELS_PATH = DATA_DIR / "F01_202w1_labels.TIF"

requires_real_data = pytest.mark.skipif(
    not (IMAGE_PATH.exists() and LABELS_PATH.exists()),
    reason="real F01_202w1 test images are not available",
)


def load_real_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the real image/labels and build a per-cell measurement table."""
    image = tifffile.imread(IMAGE_PATH)
    labels = tifffile.imread(LABELS_PATH).astype(np.int64)
    # per-label area and mean intensity in a single pass via bincount
    area = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=image.ravel().astype(float))
    cell_ids = np.nonzero(area)[0]
    cell_ids = cell_ids[cell_ids > 0]  # drop background (label 0)
    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "mean_intensity": sums[cell_ids] / area[cell_ids],
            "area": area[cell_ids],
            "size_class": np.where(
                area[cell_ids] > np.median(area[cell_ids]), "large", "small"
            ),
        }
    )
    return image, labels, df


def make_synthetic_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a small controlled image, 3-label mask and measurement table."""
    image = np.linspace(0.0, 1.0, 36).reshape(6, 6)
    mask = np.zeros((6, 6), dtype=int)
    mask[0:2, 0:2] = 1
    mask[2:4, 2:4] = 2
    mask[4:6, 4:6] = 3
    df = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [0.1, 0.5, 0.9],
        }
    )
    return image, mask, df


# --------------------------------------------------------------------------- #
# helpers: _is_categorical
# --------------------------------------------------------------------------- #
def test_is_categorical_detects_object():
    """Object/string columns are treated as categorical."""
    assert _is_categorical(pd.Series(["a", "b", "c"]))


def test_is_categorical_detects_categorical_dtype():
    """Pandas categorical-dtype columns are treated as categorical."""
    assert _is_categorical(pd.Series(["a", "b"], dtype="category"))


def test_is_categorical_false_for_numeric():
    """Integer and float columns are treated as continuous, not categorical."""
    assert not _is_categorical(pd.Series([1, 2, 3]))
    assert not _is_categorical(pd.Series([0.1, 0.2, 0.3]))


# --------------------------------------------------------------------------- #
# helpers: _categorical_palette
# --------------------------------------------------------------------------- #
def test_categorical_palette_listed_colormap():
    """A listed colormap (tab10) returns its first n colors as a list."""
    palette = _categorical_palette("tab10", 4)
    assert isinstance(palette, list)
    assert palette == list(plt.get_cmap("tab10").colors[:4])


def test_categorical_palette_continuous_colormap():
    """A continuous colormap (viridis) is sampled into n distinct colors."""
    palette = _categorical_palette("viridis", 3)
    assert len(palette) == 3
    assert len({tuple(c) for c in palette}) == 3


def test_categorical_palette_single_category():
    """A single category returns one color without dividing by zero."""
    assert len(_categorical_palette("viridis", 1)) == 1


# --------------------------------------------------------------------------- #
# overlay_labels: real data integration
# --------------------------------------------------------------------------- #
@requires_real_data
def test_returns_figure_and_axes():
    """The function returns a matplotlib Figure and Axes on real data."""
    image, labels, df = load_real_data()
    fig, ax = overlay_labels(
        image,
        labels,
        df,
        id_col="cell_id",
        measurement_col="mean_intensity",
        show=False,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


@requires_real_data
def test_image_only_when_label_mask_none():
    """Without a label mask only the image is drawn (no overlay/legend/colorbar)."""
    image, _labels, _df = load_real_data()
    fig, ax = overlay_labels(image, show=False)
    assert len(ax.images) == 1
    assert ax.get_legend() is None
    assert len(fig.axes) == 1
    plt.close(fig)


@requires_real_data
def test_segmentation_mode_labels_every_cell():
    """With no dataframe, every labeled cell is annotated with its id."""
    image, labels, _df = load_real_data()
    n_cells = len(np.unique(labels[labels > 0]))
    fig, ax = overlay_labels(image, labels, show=False)
    assert len(ax.texts) == n_cells
    assert len(ax.images) == 2  # base image + overlay
    assert ax.get_title() == "Segmentation overlay"
    plt.close(fig)


@requires_real_data
def test_continuous_measurement_adds_colorbar():
    """A continuous measurement draws a colorbar (extra axes) and no legend."""
    image, labels, df = load_real_data()
    fig, ax = overlay_labels(
        image,
        labels,
        df,
        id_col="cell_id",
        measurement_col="mean_intensity",
        show=False,
    )
    assert ax.get_legend() is None
    assert len(fig.axes) == 2
    assert ax.get_title() == "mean_intensity"
    plt.close(fig)


@requires_real_data
def test_categorical_measurement_adds_legend():
    """A categorical measurement draws a legend titled by the column."""
    image, labels, df = load_real_data()
    fig, ax = overlay_labels(
        image,
        labels,
        df,
        id_col="cell_id",
        measurement_col="size_class",
        show=False,
    )
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"large", "small"}
    assert ax.get_title() == "size_class"
    plt.close(fig)


# --------------------------------------------------------------------------- #
# overlay_labels: controlled synthetic checks
# --------------------------------------------------------------------------- #
def test_rgb_image_input():
    """A 3D RGB image is accepted and displayed."""
    rgb = np.random.default_rng(1).random((6, 6, 3))
    fig, ax = overlay_labels(rgb, show=False)
    assert isinstance(fig, Figure)
    assert len(ax.images) == 1
    plt.close(fig)


def test_constant_image_does_not_raise():
    """A constant image (max == min) is handled by the epsilon guard."""
    fig, _ax = overlay_labels(np.full((6, 6), 5.0), show=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_segmentation_mode_is_reproducible():
    """The fixed random seed makes the overlay colors reproducible."""
    image, mask, _df = make_synthetic_data()
    _fig1, ax1 = overlay_labels(image, mask, show=False)
    _fig2, ax2 = overlay_labels(image, mask, show=False)
    assert np.array_equal(ax1.images[1].get_array(), ax2.images[1].get_array())
    plt.close("all")


def test_alpha_sets_overlay_opacity():
    """The alpha argument controls the overlay opacity of labeled pixels."""
    image, mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(image, mask, alpha=0.7, show=False)
    overlay = ax.images[1].get_array()
    assert overlay[0, 5, 3] == 0.0  # background stays transparent
    assert overlay[0, 0, 3] == pytest.approx(0.7)  # labeled pixel uses alpha
    plt.close(fig)


def test_categorical_dtype_column_produces_legend():
    """A pandas categorical-dtype column also produces a legend."""
    image, mask, df = make_synthetic_data()
    df["category"] = df["category"].astype("category")
    fig, ax = overlay_labels(
        image, mask, df, id_col="label", measurement_col="category", show=False
    )
    assert ax.get_legend() is not None
    plt.close(fig)


def test_categorical_same_category_same_color():
    """Cells sharing a category get the same overlay color; others differ."""
    image, mask, df = make_synthetic_data()
    fig, ax = overlay_labels(
        image, mask, df, id_col="label", measurement_col="category", show=False
    )
    overlay = ax.images[1].get_array()
    assert np.array_equal(overlay[0, 0], overlay[5, 5])  # labels 1 and 3 -> "A"
    assert not np.array_equal(overlay[0, 0], overlay[2, 2])  # label 2 -> "B"
    plt.close(fig)


def test_continuous_respects_vmin_vmax():
    """Narrowing vmin/vmax changes the mapped colors of the overlay."""
    image, mask, df = make_synthetic_data()
    _fig1, ax1 = overlay_labels(
        image,
        mask,
        df,
        id_col="label",
        measurement_col="value",
        vmin=0.0,
        vmax=1.0,
        show=False,
    )
    _fig2, ax2 = overlay_labels(
        image,
        mask,
        df,
        id_col="label",
        measurement_col="value",
        vmin=0.4,
        vmax=0.6,
        show=False,
    )
    assert not np.array_equal(
        ax1.images[1].get_array()[0, 0], ax2.images[1].get_array()[0, 0]
    )
    plt.close("all")


def test_custom_cmap_is_accepted():
    """A custom continuous colormap is accepted."""
    image, mask, df = make_synthetic_data()
    fig, _ax = overlay_labels(
        image,
        mask,
        df,
        id_col="label",
        measurement_col="value",
        cmap="plasma",
        show=False,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# overlay_labels: show
# --------------------------------------------------------------------------- #
def test_show_defaults_to_true():
    """``plt.show`` is called by default before returning the figure."""
    image, mask, df = make_synthetic_data()
    with patch.object(plt, "show") as mock_show:
        fig, _ax = overlay_labels(
            image, mask, df, id_col="label", measurement_col="value"
        )
    assert mock_show.call_count == 1
    plt.close(fig)


def test_show_false_does_not_call_plt_show():
    """``show=False`` lets the caller control display."""
    image, mask, df = make_synthetic_data()
    with patch.object(plt, "show") as mock_show:
        fig, _ax = overlay_labels(
            image, mask, df, id_col="label", measurement_col="value", show=False
        )
    assert mock_show.call_count == 0
    plt.close(fig)


def test_show_on_image_only_path():
    """The no-overlay (image-only) figure is also shown by default."""
    image, _mask, _df = make_synthetic_data()
    with patch.object(plt, "show") as mock_show:
        fig, _ax = overlay_labels(image)
    assert mock_show.call_count == 1
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
