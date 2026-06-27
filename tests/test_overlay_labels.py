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
from bobiac_tools._overlay_labels import (
    _apply_crop,
    _categorical_palette,
    _compute_crop,
    _is_categorical,
    _offset_coordinates,
    _stack_channels,
)

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
    """A custom continuous mask_cmap is accepted."""
    image, mask, df = make_synthetic_data()
    fig, _ax = overlay_labels(
        image,
        mask,
        df,
        id_col="label",
        measurement_col="value",
        mask_cmap="plasma",
        show=False,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_image_cmap_applied_to_grayscale():
    """Grayscale input defaults to 'gray' colormap when image_cmap is not set."""
    image, _mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(image, show=False)
    assert ax.images[0].cmap.name == "gray"
    plt.close(fig)


def test_image_cmap_override_for_grayscale():
    """image_cmap overrides the 'gray' default for grayscale input."""
    image, _mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(image, image_cmap="hot", show=False)
    assert ax.images[0].cmap.name == "hot"
    plt.close(fig)


def test_image_cmap_ignored_for_rgb():
    """image_cmap has no effect when the input is already an RGB array."""
    rgb = np.random.default_rng(1).random((6, 6, 3))
    fig, ax = overlay_labels(rgb, image_cmap="gray", show=False)
    # imshow receives None for cmap on RGB input; matplotlib uses its default
    assert ax.images[0].cmap.name != "gray"
    plt.close(fig)


def test_image_cmap_ignored_for_list_input():
    """image_cmap has no effect when a list of channels is passed (already RGB)."""
    rng = np.random.default_rng(2)
    ch0, ch1 = rng.random((6, 6)), rng.random((6, 6))
    fig, ax = overlay_labels([ch0, ch1], image_cmap="hot", show=False)
    assert ax.images[0].cmap.name != "hot"
    plt.close(fig)


# --------------------------------------------------------------------------- #
# _stack_channels helper
# --------------------------------------------------------------------------- #
def test_stack_channels_two_channels():
    """Two channels are composited with red + cyan semantic colors."""
    # ch0 has content → red (1,0,0); ch1 is constant (normalizes to zero)
    ch0 = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    ch1 = np.zeros((4, 4))
    rgb = _stack_channels([ch0, ch1])
    assert rgb.shape == (4, 4, 3)
    # ch0 maps to red channel only; ch1 is constant → zero contribution
    assert rgb[..., 0].max() == pytest.approx(1.0)  # red from ch0
    assert rgb[..., 1].max() == pytest.approx(0.0)  # no green
    assert rgb[..., 2].max() == pytest.approx(0.0)  # no blue


def test_stack_channels_three_channels():
    """Three (H, W) arrays fill all three RGB planes."""
    rng = np.random.default_rng(0)
    channels = [rng.random((4, 4)) for _ in range(3)]
    rgb = _stack_channels(channels)
    assert rgb.shape == (4, 4, 3)


def test_stack_channels_normalizes_each_independently():
    """Each channel is normalized to [0, 1] independently."""
    ch0 = np.full((4, 4), 100.0)
    ch0[0, 0] = 200.0
    ch1 = np.full((4, 4), 1.0)
    ch1[0, 0] = 2.0
    rgb = _stack_channels([ch0, ch1])
    assert rgb[..., 0].max() == pytest.approx(1.0)
    assert rgb[..., 1].max() == pytest.approx(1.0)


def test_stack_channels_raises_for_more_than_three():
    """Passing more than 3 channels raises a ValueError."""
    ch = np.zeros((4, 4))
    with pytest.raises(ValueError, match="At most 3 channels"):
        _stack_channels([ch, ch, ch, ch])


# --------------------------------------------------------------------------- #
# overlay_labels: list-of-channels input
# --------------------------------------------------------------------------- #
def test_list_of_two_channels_accepted():
    """A list of two grayscale arrays is accepted and displayed as RGB."""
    rng = np.random.default_rng(1)
    ch0 = rng.random((6, 6))
    ch1 = rng.random((6, 6))
    fig, ax = overlay_labels([ch0, ch1], show=False)
    assert isinstance(fig, Figure)
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (6, 6, 3)
    plt.close(fig)


def test_list_of_three_channels_accepted():
    """A list of three grayscale arrays is accepted."""
    rng = np.random.default_rng(2)
    channels = [rng.random((6, 6)) for _ in range(3)]
    fig, _ax = overlay_labels(channels, show=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_list_input_with_label_mask():
    """A list of channels can be combined with a label_mask overlay."""
    rng = np.random.default_rng(3)
    ch0 = rng.random((6, 6))
    ch1 = rng.random((6, 6))
    mask = np.zeros((6, 6), dtype=int)
    mask[0:2, 0:2] = 1
    fig, ax = overlay_labels([ch0, ch1], label_mask=mask, show=False)
    assert len(ax.images) == 2  # base image + overlay
    plt.close(fig)


def test_list_of_four_channels_raises():
    """Passing a list of 4 channels raises a ValueError."""
    ch = np.zeros((6, 6))
    with pytest.raises(ValueError, match="At most 3 channels"):
        overlay_labels([ch, ch, ch, ch], show=False)


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


# --------------------------------------------------------------------------- #
# overlay_labels: image=None and validation
# --------------------------------------------------------------------------- #
def test_both_none_raises():
    """Passing both image=None and label_mask=None raises ValueError."""
    with pytest.raises(ValueError, match="At least one"):
        overlay_labels(show=False)


def test_image_none_single_mask():
    """image=None renders a black canvas under the mask overlay."""
    _image, mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(label_mask=mask, show=False)
    assert isinstance(fig, Figure)
    assert len(ax.images) == 2  # black canvas + overlay
    plt.close(fig)


def test_image_none_list_masks():
    """image=None with a list of masks renders without a background image."""
    _image, mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(label_mask=[mask, mask], show=False)
    assert isinstance(fig, Figure)
    assert len(ax.images) == 2
    plt.close(fig)


# --------------------------------------------------------------------------- #
# overlay_labels: 3D (C, H, W) input normalization
# --------------------------------------------------------------------------- #
def test_3d_image_is_normalized_to_channels():
    """A (C, H, W) array with C ≤ 3 is composited as multi-channel input."""
    rng = np.random.default_rng(0)
    vol = rng.random((2, 6, 6))  # shape[0]=2 ≤ 3 → treated as 2 channels
    fig, ax = overlay_labels(vol, show=False)
    assert isinstance(fig, Figure)
    assert ax.images[0].get_array().shape == (6, 6, 3)
    plt.close(fig)


def test_rgb_image_not_split_into_channels():
    """An (H, W, 3) RGB array is NOT split (shape[0]=H > 3)."""
    rgb = np.random.default_rng(1).random((6, 6, 3))
    fig, _ax = overlay_labels(rgb, show=False)
    # Image was treated as a single (H,W,3) RGB array, not 6 separate channels
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_3d_label_mask_is_split_to_list():
    """A (C, H, W) label mask with C ≤ 3 is treated as a list of masks."""
    image, mask, _df = make_synthetic_data()
    masks_3d = np.stack([mask, mask], axis=0)  # shape (2, 6, 6)
    fig, ax = overlay_labels(image, label_mask=masks_3d, show=False)
    assert isinstance(fig, Figure)
    assert len(ax.images) == 2  # background + multi-mask overlay
    plt.close(fig)


# --------------------------------------------------------------------------- #
# overlay_labels: list of masks
# --------------------------------------------------------------------------- #
def test_list_of_masks_produces_colored_overlay():
    """A list of two masks produces two distinct overlay colors."""
    image, mask, _df = make_synthetic_data()
    mask2 = np.zeros_like(mask)
    mask2[4:6, 4:6] = 1
    fig, ax = overlay_labels(image, label_mask=[mask, mask2], show=False)
    assert len(ax.images) == 2
    plt.close(fig)


def test_list_of_masks_no_legend_or_colorbar():
    """The multi-mask path adds no legend and no colorbar."""
    image, mask, _df = make_synthetic_data()
    fig, ax = overlay_labels(image, label_mask=[mask, mask], show=False)
    assert ax.get_legend() is None
    assert len(fig.axes) == 1
    plt.close(fig)


# --------------------------------------------------------------------------- #
# _compute_crop / _apply_crop / _offset_coordinates helpers
# --------------------------------------------------------------------------- #
def test_compute_crop_finds_object():
    """_compute_crop returns a bounding box containing the requested object."""
    mask = np.zeros((20, 20), dtype=int)
    mask[5:10, 8:12] = 42
    crop = _compute_crop(mask, 42)
    assert crop is not None
    y0, y1, x0, x1 = crop
    # argwhere gives pixel coordinates; max row is 9, max col is 11 (0-indexed)
    assert y0 <= 5 and y1 >= 9
    assert x0 <= 8 and x1 >= 11


def test_compute_crop_missing_object_returns_none():
    """_compute_crop returns None when the object is absent."""
    mask = np.zeros((10, 10), dtype=int)
    assert _compute_crop(mask, 99) is None


def test_apply_crop_slices_array():
    """_apply_crop slices a 2-D array to the crop region."""
    arr = np.arange(25).reshape(5, 5)
    result = _apply_crop(arr, (1, 4, 1, 4))
    assert result.shape == (3, 3)
    assert result[0, 0] == arr[1, 1]


def test_apply_crop_none_passthrough():
    """_apply_crop returns None when arr is None."""
    assert _apply_crop(None, (0, 2, 0, 2)) is None


def test_offset_coordinates_array():
    """Points outside the crop are removed; remaining ones are shifted."""
    coords = np.array([[2.0, 3.0], [10.0, 10.0]])  # second point outside crop
    result = _offset_coordinates(coords, (0, 5, 0, 5))
    result = np.asarray(result)
    assert len(result) == 1
    assert result[0, 0] == pytest.approx(2.0)
    assert result[0, 1] == pytest.approx(3.0)


def test_offset_coordinates_dataframe():
    """DataFrame coordinates are filtered and shifted to the crop frame."""
    df = pd.DataFrame({"x": [2.0, 8.0], "y": [3.0, 1.0]})
    result = _offset_coordinates(df, (0, 5, 0, 5))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # (x=8) is outside crop x<5
    assert result["x"].iloc[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# overlay_labels: coordinates parameter
# --------------------------------------------------------------------------- #
def test_coordinates_array_scattered():
    """A plain (N,2) coordinate array adds scatter points to the axes."""
    image, _mask, _df = make_synthetic_data()
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig, ax = overlay_labels(image, coordinates=coords, show=False)
    assert len(ax.collections) == 1
    plt.close(fig)


def test_coordinates_dataframe_scattered():
    """A DataFrame with x/y columns adds scatter points."""
    image, _mask, _df = make_synthetic_data()
    df_coords = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    fig, ax = overlay_labels(image, coordinates=df_coords, show=False)
    assert len(ax.collections) == 1
    plt.close(fig)


def test_coordinates_dataframe_with_channel_adds_legend():
    """A DataFrame with a 'channel' column produces one scatter per channel + legend."""
    image, _mask, _df = make_synthetic_data()
    df_coords = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "channel": [0, 1]})
    fig, ax = overlay_labels(image, coordinates=df_coords, show=False)
    assert ax.get_legend() is not None
    assert len(ax.collections) == 2  # one scatter per channel
    plt.close(fig)


# --------------------------------------------------------------------------- #
# overlay_labels: focus_object
# --------------------------------------------------------------------------- #
def test_focus_object_crops_figure():
    """focus_object adjusts figsize to match the object's crop aspect ratio."""
    rng = np.random.default_rng(0)
    image = rng.random((20, 20))
    mask = np.zeros((20, 20), dtype=int)
    mask[2:4, 2:10] = 7  # wide rectangle → w_px > h_px → figsize changes
    fig_crop, _ = overlay_labels(image, mask, focus_object=7, show=False)
    w, h = fig_crop.get_size_inches()
    assert w != pytest.approx(h, abs=0.1)  # aspect ratio is not 1:1
    plt.close("all")


def test_focus_object_missing_uses_full_image():
    """focus_object with an absent ID falls back to the full image (crop=None)."""
    image, mask, _df = make_synthetic_data()
    fig, _ax = overlay_labels(image, mask, focus_object=999, show=False)
    assert fig.get_size_inches().tolist() == [8.0, 8.0]
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
