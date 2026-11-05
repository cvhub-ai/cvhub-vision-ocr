import numpy as np
import pytest

from src.ocr.preprocess.image_processor import Preprocessor


@pytest.fixture
def sample_image() -> np.ndarray:
    """256x256 BGR image with random noise."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def pre_processor(sample_image) -> Preprocessor:
    p = Preprocessor()
    p.set_image(sample_image)
    return p


class TestInit:
    def test_accepts_numpy_array(self, pre_processor, sample_image):
        assert pre_processor.result().shape == sample_image.shape

    def test_original_not_modified(self, pre_processor, sample_image):
        orginial = sample_image.copy()
        pre_processor.gaussian_denoise()
        np.testing.assert_array_equal(sample_image, orginial)


class TestDenoise:
    def test_gaussian_denoise_changes_image(self, pre_processor, sample_image):
        result = pre_processor.gaussian_denoise().result()
        assert not np.array_equal(sample_image, result)

    def test_gaussian_denoise_preserves_shape_and_dtype(
        self, pre_processor, sample_image
    ):
        result = pre_processor.gaussian_denoise().result()
        assert sample_image.shape == result.shape
        assert sample_image.dtype == result.dtype

    def test_gaussian_denoise_records_history(self, pre_processor):
        p = pre_processor.gaussian_denoise(kernel=3)
        assert any("GaussianDenoise" in entry for entry in p.history)

    def test_nlm_denoise_changes_image(self, pre_processor, sample_image):
        result = pre_processor.nlm_denoise().result()
        assert not np.array_equal(sample_image, result)

    def test_nlm_denoise_preserves_shape_and_dtype(self, pre_processor, sample_image):
        result = pre_processor.nlm_denoise().result()
        assert sample_image.shape == result.shape
        assert sample_image.dtype == result.dtype

    def test_nlm_denoise_records_history(self, pre_processor):
        p = pre_processor.nlm_denoise(h=10, template_window=7, search_window=21)
        assert any("NLMDenoise" in entry for entry in p.history)


class TestSharpness:
    def test_unsharp_mask_changes_image(self, pre_processor, sample_image):
        result = pre_processor.unsharp_mask().result()
        assert not np.array_equal(sample_image, result)

    def test_unsharp_mask_preserves_shape_and_dtype(self, pre_processor, sample_image):
        result = pre_processor.unsharp_mask().result()
        assert sample_image.shape == result.shape
        assert sample_image.dtype == result.dtype

    def test_unsharp_mask_records_history(self, pre_processor):
        p = pre_processor.unsharp_mask()
        assert any("UnsharpMask" in entry for entry in p.history)

    def test_laplacian_sharpen_changes_image(self, pre_processor, sample_image):
        result = pre_processor.laplacian_sharpen().result()
        assert not np.array_equal(sample_image, result)

    def test_laplacian_sharpen_preserves_shape_and_dtype(
        self, pre_processor, sample_image
    ):
        result = pre_processor.laplacian_sharpen().result()
        assert sample_image.shape == result.shape
        assert sample_image.dtype == result.dtype

    def test_laplacian_sharpen_records_history(self, pre_processor):
        p = pre_processor.laplacian_sharpen()
        assert any("LaplacianSharpen" in entry for entry in p.history)


class TestChaining:
    def test_chain_returns_preprocessor(self, pre_processor):
        p = pre_processor.gaussian_denoise().unsharp_mask()
        assert isinstance(p, Preprocessor)

    def test_chain_records_all_steps(self, pre_processor):
        p = pre_processor.gaussian_denoise().unsharp_mask().laplacian_sharpen()
        assert len(p.history) == 3
        assert "GaussianDenoise" in p.history[0]
        assert "UnsharpMask" in p.history[1]
        assert "LaplacianSharpen" in p.history[2]

    def test_chain_result_differs_from_original(self, pre_processor):
        result = pre_processor.gaussian_denoise().unsharp_mask().result()
        assert not np.array_equal(result, pre_processor)


class TestHistory:
    def test_history_empty_on_init(self, pre_processor):
        p = pre_processor
        assert p.history == []

    def test_history_is_copy(self, pre_processor):
        p = pre_processor.gaussian_denoise()
        p.history.append("hack")
        assert len(p.history) == 1  # internal list unchanged


class TestResult:
    def test_result_is_copy(self, pre_processor):
        r = pre_processor.result()
        r[:] = 0  # modify the returned copy
        assert not np.array_equal(pre_processor.result(), r)  # internal unchanged

    def test_result_dtype_is_uint8(self, pre_processor):
        result = pre_processor.gaussian_denoise().result()
        assert result.dtype == np.uint8
