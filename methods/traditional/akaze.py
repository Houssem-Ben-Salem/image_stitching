import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from ..base import ImageStitchingMethod

class AKAZEStitching(ImageStitchingMethod):
    """
    Image stitching using AKAZE features and RANSAC homography estimation.
    AKAZE uses non-linear scale space for better accuracy while maintaining good speed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AKAZE stitching method.
        
        Args:
            config: Dictionary containing configuration parameters:
                - descriptor_type: Type of descriptor (default: DESCRIPTOR_MLDB)
                - descriptor_size: Size of the descriptor in bits (default: 0)
                - descriptor_channels: Number of channels in descriptor (default: 3)
                - threshold: Detector threshold (default: 0.001)
                - n_octaves: Maximum octave evolution (default: 4)
                - n_octave_layers: Number of sublevels per octave (default: 4)
                - diffusivity: Diffusivity type (default: DIFF_PM_G2)
                - hamming_thresh: Maximum allowed Hamming distance (default: 40)
                - ransac_thresh: RANSAC threshold (default: 3.0)
                - min_matches: Minimum number of matches required (default: 10)
        """
        super().__init__(config)
        
        # Initialize AKAZE detector with config parameters
        self.akaze = cv2.AKAZE_create(
            descriptor_type=config.get('descriptor_type', cv2.AKAZE_DESCRIPTOR_MLDB),
            descriptor_size=config.get('descriptor_size', 0),
            descriptor_channels=config.get('descriptor_channels', 3),
            threshold=config.get('threshold', 0.001),
            nOctaves=config.get('n_octaves', 4),
            nOctaveLayers=config.get('n_octave_layers', 4),
            diffusivity=config.get('diffusivity', cv2.KAZE_DIFF_PM_G2)
        )
        
        # Store other parameters
        self.hamming_thresh = config.get('hamming_thresh', 40)
        self.ransac_thresh = config.get('ransac_thresh', 3.0)
        self.min_matches = config.get('min_matches', 10)
        
        # Initialize Brute Force Matcher with Hamming distance
        # AKAZE uses binary descriptors like ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect AKAZE features and match them between two images.
        
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
        kpts1, desc1 = self.akaze.detectAndCompute(img1_gray, None)
        kpts2, desc2 = self.akaze.detectAndCompute(img2_gray, None)
        
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
            
        Returns:Example 
            H: 3x3 homography matrix
        """
        # Estimate homography using RANSAC
        H, mask = cv2.findHomography(kpts1, kpts2, 
                                   method=cv2.RANSAC,
                                   ransacReprojThreshold=self.ransac_thresh,
                                   confidence=0.999)  # High confidence for better results
        
        if H is None:
            raise ValueError("Failed to estimate homography")
            
        # Store inlier ratio in metadata
        self.inlier_ratio = np.sum(mask) / len(mask)
        # Store number of inliers
        self.num_inliers = np.sum(mask)
        
        return H
    
    def get_inlier_ratio(self) -> float:
        """Get the inlier ratio from the last RANSAC operation."""
        return getattr(self, 'inlier_ratio', 0.0)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the stitching process."""
        return {
            'inlier_ratio': getattr(self, 'inlier_ratio', None),
            'num_inliers': getattr(self, 'num_inliers', None),
            'akaze_params': {
                'descriptor_type': self.akaze.getDescriptorType(),
                'descriptor_size': self.akaze.getDescriptorSize(),
                'descriptor_channels': self.akaze.getDescriptorChannels(),
                'threshold': self.akaze.getThreshold(),
                'n_octaves': self.akaze.getNOctaves(),
                'n_octave_layers': self.akaze.getNOctaveLayers(),
                'diffusivity': self.akaze.getDiffusivity()
            },
            'matching_params': {
                'hamming_thresh': self.hamming_thresh,
                'ransac_thresh': self.ransac_thresh,
                'min_matches': self.min_matches
            }
        }

# configuration
default_config = {
    'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,
    'descriptor_size': 0,  # 0 means automatic
    'descriptor_channels': 3,
    'threshold': 0.001,
    'n_octaves': 4,
    'n_octave_layers': 4,
    'diffusivity': cv2.KAZE_DIFF_PM_G2,
    'hamming_thresh': 40,
    'ransac_thresh': 3.0,
    'min_matches': 10
}