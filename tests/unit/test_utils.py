import numpy as np
import pytest

from src.shared.utils.utils import Utils


@pytest.fixture
def sample_image() -> np.ndarray:
    """512x512 BGR image with gradient so ROI content is non-uniform."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        image[i, :] = i // 2
    return image


@pytest.fixture
def axis_aligned_corners() -> list[list[int]]:
    """A perfectly axis-aligned rectangle (no rotation)."""
    return [[100, 100], [300, 100], [300, 200], [100, 200]]


@pytest.fixture
def rotated_corners() -> list[list[int]]:
    """A slightly rotated quadrilateral."""
    return [[110, 100], [310, 105], [305, 205], [105, 200]]


# ------------------------------------------------------------------ #
#  sort_bbox_points
# ------------------------------------------------------------------ #
class TestSortBboxPoints:
    def test_already_sorted_unchanged(self, axis_aligned_corners):
        points = np.array(axis_aligned_corners, dtype=np.float32)
        sorted_points = Utils.sort_bbox_points(points)
        assert sorted_points.shape == (4, 2)
        assert sorted_points[0][0] < sorted_points[1][0]  # top-left x < top-right x
        assert sorted_points[0][1] < sorted_points[3][1]  # top-left y < bottom-left y
        assert sorted_points[1][0] > sorted_points[0][0]  # top-right x > top-left x
        assert sorted_points[2][1] > sorted_points[1][1]  # bottom-right y > top-right y

    def test_shuffled_gives_same_result(self, axis_aligned_corners):
        points = np.array(axis_aligned_corners, dtype=np.float32)
        shuffled = points[[2, 0, 3, 1]]
        sorted_points = Utils.sort_bbox_points(shuffled)
        # all 4 points should still be present
        for pt in points:
            assert any(np.allclose(pt, sp) for sp in sorted_points)


# ------------------------------------------------------------------ #
#  extract_text_roi
# ------------------------------------------------------------------ #
class TestExtractTextRoi:
    def test_output_is_numpy_array(self, sample_image, axis_aligned_corners):
        roi = Utils.extract_text_roi(sample_image, axis_aligned_corners)
        assert isinstance(roi, np.ndarray)

    def test_output_dtype_is_uint8(self, sample_image, axis_aligned_corners):
        roi = Utils.extract_text_roi(sample_image, axis_aligned_corners)
        assert roi.dtype == np.uint8

    def test_output_is_3_channel(self, sample_image, axis_aligned_corners):
        roi = Utils.extract_text_roi(sample_image, axis_aligned_corners)
        assert len(roi.shape) == 3 and roi.shape[2] == 3

    def test_output_shape_matches_bbox_dimensions(
        self, sample_image, axis_aligned_corners
    ):
        roi = Utils.extract_text_roi(sample_image, axis_aligned_corners)
        # bbox is 200x100, allow ±2px tolerance from float rounding
        assert abs(roi.shape[1] - 200) <= 2  # width
        assert abs(roi.shape[0] - 100) <= 2  # height

    def test_shuffled_corners_same_shape(self, sample_image, axis_aligned_corners):
        """Point order should not affect output dimensions."""
        corners_shuffled = [axis_aligned_corners[i] for i in [2, 0, 3, 1]]
        roi_original = Utils.extract_text_roi(sample_image, axis_aligned_corners)
        roi_shuffled = Utils.extract_text_roi(sample_image, corners_shuffled)
        assert roi_original.shape == roi_shuffled.shape

    def test_rotated_bbox_produces_valid_roi(self, sample_image, rotated_corners):
        roi = Utils.extract_text_roi(sample_image, rotated_corners)
        assert roi.size > 0
        assert roi.dtype == np.uint8

    def test_original_image_not_modified(self, sample_image, axis_aligned_corners):
        original = sample_image.copy()
        Utils.extract_text_roi(sample_image, axis_aligned_corners)
        np.testing.assert_array_equal(sample_image, original)
