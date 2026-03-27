"""Unit tests for the image / PDF preprocessor."""
import io
import pytest
from PIL import Image
import numpy as np

from ocr_model.preprocessor import (
    _pil_to_cv,
    _cv_to_pil,
    _to_grayscale,
    _clahe_enhance,
    _deskew,
    _upscale_if_needed,
    _binarize,
    preprocess_image,
)


def _make_white_image(w: int = 200, h: int = 200) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8), mode="RGB")


def _make_gray_image(w: int = 200, h: int = 200) -> Image.Image:
    return Image.fromarray(np.full((h, w), 128, dtype=np.uint8), mode="L")


class TestConversions:
    def test_pil_to_cv_shape(self):
        pil = _make_white_image(100, 80)
        arr = _pil_to_cv(pil)
        assert arr.shape == (80, 100, 3)

    def test_cv_to_pil_mode(self):
        import cv2, numpy as np
        arr = np.zeros((80, 100, 3), dtype=np.uint8)
        pil = _cv_to_pil(arr)
        assert pil.mode == "RGB"
        assert pil.size == (100, 80)

    def test_roundtrip(self):
        pil = _make_white_image(60, 40)
        arr = _pil_to_cv(pil)
        back = _cv_to_pil(arr)
        assert back.size == pil.size


class TestGrayscale:
    def test_rgb_to_gray(self):
        import numpy as np, cv2
        arr = np.full((50, 50, 3), 200, dtype=np.uint8)
        gray = _to_grayscale(arr)
        assert gray.ndim == 2

    def test_already_gray_passthrough(self):
        import numpy as np
        arr = np.full((50, 50), 100, dtype=np.uint8)
        out = _to_grayscale(arr)
        assert out is arr


class TestClahe:
    def test_output_shape_preserved(self):
        import numpy as np
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        enhanced = _clahe_enhance(gray)
        assert enhanced.shape == gray.shape

    def test_output_dtype(self):
        import numpy as np
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        enhanced = _clahe_enhance(gray)
        assert enhanced.dtype == np.uint8


class TestUpscale:
    def test_small_image_is_upscaled(self):
        import numpy as np
        gray = np.zeros((200, 300), dtype=np.uint8)
        out = _upscale_if_needed(gray, min_height=500)
        assert out.shape[0] >= 500

    def test_large_image_unchanged(self):
        import numpy as np
        gray = np.zeros((1200, 800), dtype=np.uint8)
        out = _upscale_if_needed(gray, min_height=500)
        assert out.shape[0] == 1200


class TestBinarize:
    def test_output_is_binary(self):
        import numpy as np
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        binary = _binarize(gray)
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255})


class TestPreprocessImage:
    def test_returns_pil_image(self, tmp_path):
        pil = _make_white_image(300, 300)
        path = tmp_path / "test.jpg"
        pil.save(str(path))
        result = preprocess_image(path)
        assert isinstance(result, Image.Image)

    def test_accepts_pil_directly(self):
        pil = _make_white_image(300, 300)
        result = preprocess_image(pil)
        assert isinstance(result, Image.Image)

    def test_unsupported_extension_raises(self, tmp_path):
        from ocr_model.preprocessor import load_pages
        path = tmp_path / "report.xyz"
        path.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_pages(path)
