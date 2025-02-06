import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.traditional.sift import SIFTStitching, default_config
from utils.data_loader import HPatchesDataset

class TestSIFTStitching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across all tests."""
        cls.method = SIFTStitching(default_config)
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
        print(f"Number of matches found: {len(matches)}")
        
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
        print(f"\nStitching results:")
        print(f"- Execution time: {metadata['execution_time']:.3f} seconds")
        print(f"- Number of matches: {metadata['num_matches']}")
        print(f"- Output image shape: {result.shape}")
        print(f"- Sequence: {self.test_pair.sequence_name}")

if __name__ == '__main__':
    unittest.main(verbosity=2)