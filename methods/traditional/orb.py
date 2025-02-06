import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from ..base import ImageStitchingMethod

class ORBStitching(ImageStitchingMethod):
    """
    Image stitching using ORB features and RANSAC homography estimation.
    ORB is generally faster than SIFT but might be less accurate for some cases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ORB stitching method.
        
        Args:
            config: Dictionary containing configuration parameters:
                - nfeatures: Number of features to detect (default: 5000)
                - scale_factor: Pyramid scale factor (default: 1.2)
                - nlevels: Number of pyramid levels (default: 8)
                - edge_threshold: Edge threshold (default: 31)
                - first_level: First pyramid level (default: 0)
                - WTA_K: Number of points for orientation (2,3,4) (default: 2)
                - patch_size: Patch size (default: 31)
                - fast_threshold: FAST threshold (default: 20)
                - hamming_thresh: Maximum allowed Hamming distance (default: 30)
                - ransac_thresh: RANSAC threshold (default: 4.0)
                - min_matches: Minimum number of matches required (default: 10)
        """
        super().__init__(config)
        
        # Initialize ORB detector with config parameters
        self.orb = cv2.ORB_create(
            nfeatures=config.get('nfeatures', 5000),
            scaleFactor=config.get('scale_factor', 1.2),
            nlevels=config.get('nlevels', 8),
            edgeThreshold=config.get('edge_threshold', 31),
            firstLevel=config.get('first_level', 0),
            WTA_K=config.get('WTA_K', 2),
            patchSize=config.get('patch_size', 31),
            fastThreshold=config.get('fast_threshold', 20)
        )
        
        # Store other parameters
        self.hamming_thresh = config.get('hamming_thresh', 30)
        self.ransac_thresh = config.get('ransac_thresh', 4.0)
        self.min_matches = config.get('min_matches', 10)
        
        # Initialize Brute Force Matcher with Hamming distance
        # Using Hamming distance because ORB produces binary descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect ORB features and match them between two images.
        
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
        kpts1, desc1 = self.orb.detectAndCompute(img1_gray, None)
        kpts2, desc2 = self.orb.detectAndCompute(img2_gray, None)
        
        if len(kpts1) < self.min_matches or len(kpts2) < self.min_matches:
            raise ValueError(f"Not enough keypoints found: {len(kpts1)} and {len(kpts2)}")
            
        if desc1 is None or desc2 is None:
            raise ValueError("No descriptors computed")
        
        # Match descriptors
        matches = self.matcher.match(desc1, desc2)
        
        # Filter matches based on Hamming distance
        good_matches = [m for m in matches if m.distance < self.hamming_thresh]
        
        if len(good_matches) < self.min_matches:
            raise ValueError(f"Not enough good matches found: {len(good_matches)}")
        
        # Sort matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        matched_pts1 = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
        matched_pts2 = np.float32([kpts2[m.trainIdx].pt for m in good_matches])
                
        return matched_pts1, matched_pts2, np.array(good_matches)

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
    
    def get_inlier_ratio(self) -> float:
        """Get the inlier ratio from the last RANSAC operation."""
        return getattr(self, 'inlier_ratio', 0.0)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the stitching process."""
        return {
            'inlier_ratio': getattr(self, 'inlier_ratio', None),
            'orb_params': {
                'nfeatures': self.orb.getMaxFeatures(),
                'nlevels': self.orb.getNLevels(),
                'edge_threshold': self.orb.getEdgeThreshold(),
                'WTA_K': self.orb.getWTA_K(),
                'patch_size': self.orb.getPatchSize(),
                'fast_threshold': self.orb.getFastThreshold()
            },
            'matching_params': {
                'hamming_thresh': self.hamming_thresh,
                'ransac_thresh': self.ransac_thresh,
                'min_matches': self.min_matches
            }
        }

# configuration
default_config = {
    'nfeatures': 5000,
    'scale_factor': 1.2,
    'nlevels': 8,
    'edge_threshold': 31,
    'first_level': 0,
    'WTA_K': 2,
    'patch_size': 31,
    'fast_threshold': 20,
    'hamming_thresh': 30,
    'ransac_thresh': 4.0,
    'min_matches': 10
}