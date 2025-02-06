# methods/deep_learning/superpoint.py
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from ..base import ImageStitchingMethod
import sys 

class SuperPointStitching(ImageStitchingMethod):
    """
    Image stitching using SuperPoint features and SuperGlue matching.
    Integrates with pretrained SuperPoint and SuperGlue models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SuperPoint+SuperGlue stitching method.
        
        Args:
            config: Dictionary containing configuration parameters:
                - weights_path: Path to SuperGluePretrainedNetwork directory
                - superpoint: SuperPoint configuration
                    - nms_radius: Non Maximum Suppression radius
                    - keypoint_threshold: Detector confidence threshold
                    - max_keypoints: Maximum number of keypoints
                - superglue: SuperGlue configuration
                    - weights: 'indoor' or 'outdoor'
                    - sinkhorn_iterations: Number of Sinkhorn iterations
                    - match_threshold: Match confidence threshold
                - ransac_thresh: RANSAC threshold for homography estimation
                - min_matches: Minimum number of required matches
                - device: 'cuda' or 'cpu'
        """
        super().__init__(config)
        
        # Store parameters
        self.ransac_thresh = config.get('ransac_thresh', 4.0)
        self.min_matches = config.get('min_matches', 10)
        
        # Setup device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SuperPoint + SuperGlue
        weights_path = Path(config['weights_path'])
        if not weights_path.exists():
            raise ValueError(f"Weights directory not found: {weights_path}")
            
        sys.path.append(str(weights_path))
        from SuperGluePretrainedNetwork.models.matching import Matching
        
        # Configure models
        matching_config = {
            'superpoint': {
                'nms_radius': config.get('nms_radius', 4),
                'keypoint_threshold': config.get('keypoint_threshold', 0.005),
                'max_keypoints': config.get('max_keypoints', 1024)
            },
            'superglue': {
                'weights': config.get('superglue_weights', 'indoor'),
                'sinkhorn_iterations': config.get('sinkhorn_iterations', 20),
                'match_threshold': config.get('match_threshold', 0.2),
            }
        }
        
        self.matching = Matching(matching_config).eval().to(self.device)

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for SuperPoint."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize image
        img = img.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img = torch.from_numpy(img)[None, None].to(self.device)
        
        return img

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect SuperPoint features and match them using SuperGlue.
        
        Args:
            img1: First image (reference)
            img2: Second image (target)
            
        Returns:
            kpts1: Matched keypoints from first image
            kpts2: Matched keypoints from second image
            matches: Array of matches between keypoints
        """
        # Preprocess images
        inp1 = self.preprocess_image(img1)
        inp2 = self.preprocess_image(img2)
        
        # Perform matching
        with torch.no_grad():
            pred = self.matching({
                'image0': inp1,
                'image1': inp2
            })
        
        # Extract predictions
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        # Keep valid matches
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        
        if len(mkpts0) < self.min_matches:
            raise ValueError(f"Not enough matches found: {len(mkpts0)}")
            
        # Store confidence scores for metadata
        self.match_confidences = mconf
        
        return mkpts0, mkpts1, matches[valid]

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
                                   ransacReprojThreshold=self.ransac_thresh,
                                   confidence=0.999)
        
        if H is None:
            raise ValueError("Failed to estimate homography")
            
        # Store inlier ratio in metadata
        self.inlier_ratio = np.sum(mask) / len(mask)
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
            'match_confidences': getattr(self, 'match_confidences', None),
            'superpoint_config': self.matching.superpoint.config,
            'superglue_config': self.matching.superglue.config,
            'device': self.device
        }

# Example configuration
default_config = {
    'weights_path': 'SuperGluePretrainedNetwork',
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024,
    'superglue_weights': 'indoor',
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
    'ransac_thresh': 4.0,
    'min_matches': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}