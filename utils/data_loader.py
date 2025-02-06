import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass

@dataclass
class ImagePair:
    """Class to store an image pair and its metadata."""
    ref_image: np.ndarray
    target_image: np.ndarray
    homography: np.ndarray
    sequence_name: str
    target_idx: int
    is_viewpoint: bool

class HPatchesDataset:
    """Loader for HPatches dataset."""
    
    def __init__(self, root_dir: str):
        """
        Initialize the HPatches dataset loader.
        
        Args:
            root_dir: Path to the root directory containing HPatches sequences
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory {root_dir} not found")
            
        # Get all sequence directories
        self.sequences = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Separate viewpoint and illumination sequences
        self.viewpoint_sequences = [seq for seq in self.sequences if seq.name.startswith('v_')]
        self.illumination_sequences = [seq for seq in self.sequences if seq.name.startswith('i_')]
        
        print(f"Found {len(self.viewpoint_sequences)} viewpoint sequences and "
              f"{len(self.illumination_sequences)} illumination sequences")

    def load_image(self, path: Path) -> np.ndarray:
        """Load and convert an image to RGB format."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_homography(self, path: Path) -> np.ndarray:
        """Load homography matrix from file."""
        try:
            return np.loadtxt(path)
        except Exception as e:
            raise ValueError(f"Failed to load homography from {path}: {e}")

    def get_sequence_pairs(self, sequence_dir: Path) -> List[ImagePair]:
        """Get all image pairs from a sequence."""
        pairs = []
        is_viewpoint = sequence_dir.name.startswith('v_')
        
        # Load reference image (1.ppm)
        ref_image_path = sequence_dir / '1.ppm'
        ref_image = self.load_image(ref_image_path)
        
        # Load each target image and corresponding homography
        for i in range(2, 7):  # Images 2-6
            target_image_path = sequence_dir / f'{i}.ppm'
            if not target_image_path.exists():
                continue
                
            target_image = self.load_image(target_image_path)
            
            # Load homography if viewpoint sequence, else use identity
            if is_viewpoint:
                h_path = sequence_dir / f'H_1_{i}'
                homography = self.load_homography(h_path)
            else:
                homography = np.eye(3)
            
            pairs.append(ImagePair(
                ref_image=ref_image,
                target_image=target_image,
                homography=homography,
                sequence_name=sequence_dir.name,
                target_idx=i,
                is_viewpoint=is_viewpoint
            ))
            
        return pairs

    def get_pairs(self, 
                 type: str = 'all', 
                 max_pairs: Optional[int] = None) -> List[ImagePair]:
        """
        Get image pairs from the dataset.
        
        Args:
            type: Type of sequences to load ('all', 'viewpoint', or 'illumination')
            max_pairs: Maximum number of pairs to return (None for all)
            
        Returns:
            List of ImagePair objects
        """
        if type not in ['all', 'viewpoint', 'illumination']:
            raise ValueError(f"Invalid type: {type}")
            
        sequences = []
        if type in ['all', 'viewpoint']:
            sequences.extend(self.viewpoint_sequences)
        if type in ['all', 'illumination']:
            sequences.extend(self.illumination_sequences)
            
        all_pairs = []
        for seq in sequences:
            pairs = self.get_sequence_pairs(seq)
            all_pairs.extend(pairs)
            
            if max_pairs and len(all_pairs) >= max_pairs:
                all_pairs = all_pairs[:max_pairs]
                break
                
        return all_pairs

    def get_pair_generator(self, 
                         type: str = 'all', 
                         batch_size: Optional[int] = None) -> Generator[List[ImagePair], None, None]:
        """
        Get a generator that yields batches of image pairs.
        
        Args:
            type: Type of sequences to load ('all', 'viewpoint', or 'illumination')
            batch_size: Number of pairs to yield at once (None for all pairs)
            
        Yields:
            List of ImagePair objects
        """
        pairs = self.get_pairs(type)
        
        if batch_size is None:
            yield pairs
            return
            
        for i in range(0, len(pairs), batch_size):
            yield pairs[i:i + batch_size]

# Example usage
if __name__ == "__main__":
    # Example of how to use the dataset loader
    dataset = HPatchesDataset("path/to/hpatches-sequences")
    
    # Get all viewpoint pairs
    viewpoint_pairs = dataset.get_pairs(type='viewpoint', max_pairs=10)
    
    # Process pairs in batches
    for batch in dataset.get_pair_generator(type='all', batch_size=4):
        for pair in batch:
            print(f"Processing pair from sequence {pair.sequence_name}, "
                  f"target image {pair.target_idx}")
            # Your processing code here