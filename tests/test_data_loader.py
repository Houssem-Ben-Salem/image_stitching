import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import HPatchesDataset, ImagePair

class TestHPatchesDataset(unittest.TestCase):
    def setUp(self):
        # Update this path to your dataset location
        self.dataset_path = Path("hpatches-sequences-release")
        self.dataset = HPatchesDataset(self.dataset_path)

    def test_dataset_initialization(self):
        """Test if dataset is initialized correctly."""
        self.assertTrue(len(self.dataset.sequences) > 0)
        self.assertTrue(len(self.dataset.viewpoint_sequences) > 0)
        self.assertTrue(len(self.dataset.illumination_sequences) > 0)

    def test_image_loading(self):
        """Test if images are loaded correctly."""
        # Get first viewpoint sequence
        seq = self.dataset.viewpoint_sequences[0]
        image_path = seq / '1.ppm'
        image = self.dataset.load_image(image_path)
        
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(image.shape), 3)  # Should be RGB
        self.assertEqual(image.shape[2], 3)    # 3 channels

    def test_homography_loading(self):
        """Test if homography matrices are loaded correctly."""
        # Get first viewpoint sequence
        seq = self.dataset.viewpoint_sequences[0]
        h_path = seq / 'H_1_2'
        h_matrix = self.dataset.load_homography(h_path)
        
        self.assertIsInstance(h_matrix, np.ndarray)
        self.assertEqual(h_matrix.shape, (3, 3))

    def test_pair_generation(self):
        """Test if image pairs are generated correctly."""
        # Test viewpoint pairs
        viewpoint_pairs = self.dataset.get_pairs(type='viewpoint', max_pairs=5)
        self.assertEqual(len(viewpoint_pairs), 5)
        self.assertTrue(all(pair.is_viewpoint for pair in viewpoint_pairs))

        # Test illumination pairs
        illumination_pairs = self.dataset.get_pairs(type='illumination', max_pairs=5)
        self.assertEqual(len(illumination_pairs), 5)
        self.assertTrue(all(not pair.is_viewpoint for pair in illumination_pairs))

    def test_batch_generator(self):
        """Test if batch generator works correctly."""
        batch_size = 3
        generator = self.dataset.get_pair_generator(type='all', batch_size=batch_size)
        
        # Get first batch
        first_batch = next(generator)
        self.assertLessEqual(len(first_batch), batch_size)
        self.assertIsInstance(first_batch[0], ImagePair)

if __name__ == '__main__':
    unittest.main()