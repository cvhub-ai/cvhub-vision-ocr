import cv2
import numpy as np


class Preprocessor:
    def __init__(self) -> None:
        self._history: list[str] = []

    def set_image(self, image: np.ndarray) -> None:
        self._image = image.copy()

    # ------------------------------------------------------------------ #
    #  Denoise
    # ------------------------------------------------------------------ #

    def gaussian_denoise(self, kernel: int = 3) -> "Preprocessor":
        self._image = cv2.GaussianBlur(self._image, (kernel, kernel), 0)
        self._history.append(f"GaussianDenoise(kernel={kernel})")
        return self

    def nlm_denoise(
        self,
        h: float = 10,
        template_window: int = 7,
        search_window: int = 21,
    ) -> "Preprocessor":
        is_color = len(self._image.shape) == 3 and self._image.shape[2] == 3
        if is_color:
            self._image = cv2.fastNlMeansDenoisingColored(
                self._image, None, h, h, template_window, search_window
            )
        else:
            self._image = cv2.fastNlMeansDenoising(
                self._image, None, h, template_window, search_window
            )
        self._history.append(f"NLMDenoise(h={h})")
        return self

    # ------------------------------------------------------------------ #
    #  Sharpness
    # ------------------------------------------------------------------ #

    def unsharp_mask(self) -> "Preprocessor":
        kernel = (
            np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]], dtype=np.float32) / 9
        )
        self._image = cv2.filter2D(self._image, -1, kernel)
        self._history.append(
            "UnsharpMask(kernel=[[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]])"
        )
        return self

    def laplacian_sharpen(self) -> "Preprocessor":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) / 9
        self._image = cv2.filter2D(self._image, -1, kernel)
        self._history.append(
            "LaplacianSharpen(kernel=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]])"
        )
        return self

    # ------------------------------------------------------------------ #
    #  Tools
    # ------------------------------------------------------------------ #

    def result(self) -> np.ndarray:
        return self._image.copy()

    def clear_history(self) -> None:
        self._history = []

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def __repr__(self) -> str:
        h, w = self._image.shape[:2]
        c = self._image.shape[2] if len(self._image.shape) == 3 else 1
        ops = " -> ".join(self._history) if self._history else "nothing"
        return f"Preprocessor(size={w}x{h}, channels={c}, ops=[{ops}])"
