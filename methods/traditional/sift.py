import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from ..base import ImageStitchingMethod

class SIFTStitching(ImageStitchingMethod):
    """
    Image stitching using SIFT features and RANSAC homography estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SIFT stitching method.
        
        Args:
            config: Dictionary containing configuration parameters:
                - nfeatures: Number of features to detect (default: 0, unlimited)
                - n_octave_layers: Number of octave layers (default: 3)
                - contrast_threshold: Contrast threshold (default: 0.04)
                - edge_threshold: Edge threshold (default: 10)
                - sigma: Gaussian sigma (default: 1.6)
                - ratio_thresh: Lowe's ratio test threshold (default: 0.75)
                - ransac_thresh: RANSAC threshold (default: 4.0)
                - min_matches: Minimum number of matches required (default: 10)
        """
        super().__init__(config)
        
        # Initialize SIFT detector with config parameters
        self.sift = cv2.SIFT_create(
            nfeatures=config.get('nfeatures', 0),
            nOctaveLayers=config.get('n_octave_layers', 3),
            contrastThreshold=config.get('contrast_threshold', 0.04),
            edgeThreshold=config.get('edge_threshold', 10),
            sigma=config.get('sigma', 1.6)
        )
        
        # Store other parameters
        self.ratio_thresh = config.get('ratio_thresh', 0.75)
        self.ransac_thresh = config.get('ransac_thresh', 4.0)
        self.min_matches = config.get('min_matches', 10)
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def get_inlier_ratio(self) -> float:
        """Get the inlier ratio from the last RANSAC operation."""
        return getattr(self, 'inlier_ratio', 0.0)

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect SIFT features and match them between two images.
        
        Args:
            img1: First image (reference)
            img2: Second image (target)
            
        Returns:
            kpts1: Matched keypoints from first image
            kpts2: Matched keypoints from second image
            matches: Array of matches between keypoints
        """
        # Convert images to grayscale if they're in color
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Detect keypoints and compute descriptors
        kpts1, desc1 = self.sift.detectAndCompute(img1_gray, None)
        kpts2, desc2 = self.sift.detectAndCompute(img2_gray, None)
        
        if len(kpts1) < self.min_matches or len(kpts2) < self.min_matches:
            raise ValueError(f"Not enough keypoints found: {len(kpts1)} and {len(kpts2)}")
            
        if desc1 is None or desc2 is None:
            raise ValueError("No descriptors computed")
        
        # Match descriptors using Lowe's ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        matched_pts1 = []
        matched_pts2 = []
        
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)
                matched_pts1.append(kpts1[m.queryIdx].pt)
                matched_pts2.append(kpts2[m.trainIdx].pt)
                
        if len(good_matches) < self.min_matches:
            raise ValueError(f"Not enough good matches found: {len(good_matches)}")
            
        return (np.float32(matched_pts1), 
                np.float32(matched_pts2), 
                np.array(good_matches))

    def estimate_homography(self, kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray) -> np.ndarray:
        """
        Estimate homography matrix using RANSAC.
        
        Args:
            kpts1: Matched keypoints from first image
            kpts2: Matched keypoints from second image
            matches: Array of matches between keypoints
            
        Returns:
            H: 3x3 homography matrix
        """
        # Estimate homography using RANSAC
        H, mask = cv2.findHomography(kpts1, kpts2, 
                                   method=cv2.RANSAC,
                                   ransacReprojThreshold=self.ransac_thresh)
        
        if H is None:
            raise ValueError("Failed to estimate homography")
            
        # Store inlier ratio in metadata
        self.inlier_ratio = np.sum(mask) / len(mask)
        
        return H
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the stitching process."""
        return {
            'inlier_ratio': getattr(self, 'inlier_ratio', None),
            'sift_params': {
                'nfeatures': self.sift.getNFeatures(),
                'n_octave_layers': self.sift.getNOctaveLayers(),
                'contrast_threshold': self.sift.getContrastThreshold(),
                'edge_threshold': self.sift.getEdgeThreshold(),
                'sigma': self.sift.getSigma()
            },
            'matching_params': {
                'ratio_thresh': self.ratio_thresh,
                'ransac_thresh': self.ransac_thresh,
                'min_matches': self.min_matches
            }
        }

#configuration
default_config = {
    'nfeatures': 0,  # 0 means unlimited
    'n_octave_layers': 3,
    'contrast_threshold': 0.04,
    'edge_threshold': 10,
    'sigma': 1.6,
    'ratio_thresh': 0.75,
    'ransac_thresh': 4.0,
    'min_matches': 10
}