import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.deep_learning.superpoint import SuperPointStitching, default_config
from utils.data_loader import HPatchesDataset

class TestSuperPointStitching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across all tests."""
        config = default_config.copy()
        config['weights_path'] = 'SuperGluePretrainedNetwork/models/weights/superglue_indoor.pth'  
        cls.method = SuperPointStitching(config)
        cls.dataset = HPatchesDataset("hpatches-sequences-release")
        # Get a single pair that we'll reuse for all tests
        cls.test_pairs = cls.dataset.get_pairs(type='viewpoint', max_pairs=1)
        cls.test_pair = cls.test_pairs[0]
    
    def test_feature_detection(self):
        """Test if feature detection works."""
        # Detect and match features
        kpts1, kpts2, matches = self.method.detect_and_match_features(
            self.test_pair.ref_image, self.test_pair.target_image
        )
        
        # Check outputs
        self.assertIsInstance(kpts1, np.ndarray)
        self.assertIsInstance(kpts2, np.ndarray)
        self.assertIsInstance(matches, np.ndarray)
        self.assertTrue(len(matches) >= self.method.min_matches)
        print(f"Number of SuperPoint matches found: {len(matches)}")
        print(f"Average match confidence: {self.method.match_confidences.mean():.3f}")
        
    def test_homography_estimation(self):
        """Test if homography estimation works."""
        # Get features and estimate homography
        kpts1, kpts2, matches = self.method.detect_and_match_features(
            self.test_pair.ref_image, self.test_pair.target_image
        )
        H = self.method.estimate_homography(kpts1, kpts2, matches)
        
        # Check homography matrix
        self.assertIsInstance(H, np.ndarray)
        self.assertEqual(H.shape, (3, 3))
        print(f"Estimated homography matrix:\n{H}")
        print(f"Number of inliers: {self.method.num_inliers}")
        print(f"Inlier ratio: {self.method.inlier_ratio:.3f}")
        
    def test_full_stitching(self):
        """Test the complete stitching pipeline."""
        # Perform stitching
        result, metadata = self.method.stitch_images(
            self.test_pair.ref_image, self.test_pair.target_image
        )
        
        # Check results
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(len(result.shape) == 3)  # Should be RGB
        self.assertGreater(result.shape[0], 0)   # Should have height
        self.assertGreater(result.shape[1], 0)   # Should have width
        
        # Print some interesting metadata
        print(f"\nSuperPoint+SuperGlue Stitching results:")
        print(f"- Execution time: {metadata['execution_time']:.3f} seconds")
        print(f"- Number of matches: {metadata['num_matches']}")
        
        # Safely print inliers and ratio
        inliers = metadata.get('num_inliers', 'N/A')
        print(f"- Number of inliers: {inliers}")
        
        ratio = metadata.get('inlier_ratio')
        if isinstance(ratio, (float, np.float32, np.float64)):
            print(f"- Inlier ratio: {ratio:.3f}")
        else:
            print(f"- Inlier ratio: N/A")
            
        # Safely print match confidence
        confidences = metadata.get('match_confidences')
        if confidences is not None:
            print(f"- Average match confidence: {confidences.mean():.3f}")
        else:
            print("- Average match confidence: N/A")
            
        print(f"- Output image shape: {result.shape}")
        print(f"- Sequence: {self.test_pair.sequence_name}")
        
        # Compare with ground truth homography if available
        if hasattr(self.test_pair, 'homography'):
            print("\nGround truth comparison:")
            error = np.linalg.norm(metadata['homography'] - self.test_pair.homography)
            print(f"- Homography error: {error:.3f}")

if __name__ == '__main__':
    unittest.main(verbosity=2)