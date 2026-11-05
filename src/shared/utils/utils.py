import cv2
import numpy as np


class Utils:
    @staticmethod
    def sort_bbox_points(pts: np.ndarray) -> np.ndarray:
        """
        Ensure point order is: top-left → top-right → bottom-right → bottom-left.
        """
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    @staticmethod
    def extract_text_roi(image: np.ndarray, corners: list[list[int]]) -> np.ndarray:
        """
        Extract and rectify a text region from the image using perspective transform.

        Applies sort_bbox_points to ensure correct point order before transforming,
        so corners do not need to be pre-sorted by the caller.

        Args:
            image:   Input image in BGR format (H, W, C).
            corners: Four corner points of the text region [[x1,y1], ..., [x4,y4]].

        Returns:
            Rectified text ROI as a BGR numpy array.
        """
        points = np.array(corners, dtype=np.float32)
        points = Utils.sort_bbox_points(points)

        width = max(
            np.linalg.norm(points[1] - points[0]),  # top
            np.linalg.norm(points[2] - points[3]),  # bottom
        )
        height = max(
            np.linalg.norm(points[3] - points[0]),  # left
            np.linalg.norm(points[2] - points[1]),  # right
        )
        width, height = int(width), int(height)

        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(points, dst)
        text_roi = cv2.warpPerspective(image, M, (width, height))

        return text_roi
