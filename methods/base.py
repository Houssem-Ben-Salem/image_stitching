from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class ImageStitchingMethod(ABC):
    """Base class for all image stitching methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image stitching method.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect and match features between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            kpts1: Keypoints in first image
            kpts2: Keypoints in second image
            matches: Feature matches between images
        """
        pass
    
    @abstractmethod
    def estimate_homography(self, kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray) -> np.ndarray:
        """
        Estimate homography matrix from matched features.
        
        Args:
            kpts1: Keypoints from first image
            kpts2: Keypoints from second image
            matches: Feature matches between images
            
        Returns:
            H: 3x3 homography matrix
        """
        pass
    
    def stitch_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stitch two images together.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            stitched_img: Stitched image
            metadata: Dictionary containing process metadata (timing, matches, etc.)
        """
        # Time the process
        start_time = cv2.getTickCount()
        
        # Detect and match features
        kpts1, kpts2, matches = self.detect_and_match_features(img1, img2)
        
        # Estimate homography
        H = self.estimate_homography(kpts1, kpts2, matches)
        
        # Warp and blend images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create output canvas
        points1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        points2 = cv2.perspectiveTransform(points1, H)
        points = np.concatenate((points1, points2), axis=0)
        
        [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
        
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]], 
                                [0, 1, translation_dist[1]], 
                                [0, 0, 1]])
        
        output_img = cv2.warpPerspective(img1, H_translation.dot(H),
                                       (xmax-xmin, ymax-ymin))
        output_img[-ymin:h1-ymin, -xmin:w1-xmin] = img2
        
        # Calculate execution time
        exec_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        metadata = {
            'execution_time': exec_time,
            'num_matches': len(matches),
            'homography': H,
            'keypoints1': kpts1,
            'keypoints2': kpts2,
            'matches': matches
        }
        
        return output_img, metadata