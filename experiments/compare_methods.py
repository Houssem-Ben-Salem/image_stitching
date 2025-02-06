import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.gridspec as gridspec
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class StitchingVisualizer:
    def __init__(self, methods: Dict[str, Any], output_dir: str = 'visualization_results'):
        """
        Initialize visualizer.
        
        Args:
            methods: Dictionary of method names and their instances
            output_dir: Directory to save visualization results
        """
        self.methods = methods
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up colors for different methods
        self.colors = {
            'SIFT': (0, 255, 0),      # Green
            'ORB': (255, 0, 0),       # Blue
            'AKAZE': (0, 0, 255),     # Red
            'SuperPoint': (255, 0, 255)  # Magenta
        }

    def draw_matches(self, img1: np.ndarray, img2: np.ndarray, 
                kpts1: np.ndarray, kpts2: np.ndarray, 
                matches: np.ndarray, color: tuple) -> np.ndarray:
        """Draw matches between two images."""
        try:
            # Convert keypoints to cv2.KeyPoint format if they're numpy arrays
            if isinstance(kpts1, np.ndarray):
                kpts1_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts1]
            else:
                kpts1_cv = kpts1
                
            if isinstance(kpts2, np.ndarray):
                kpts2_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts2]
            else:
                kpts2_cv = kpts2
            
            # Handle different match formats and validate indices
            if isinstance(matches[0], cv2.DMatch):
                # Filter valid matches
                good_matches = [m for m in matches 
                            if 0 <= m.queryIdx < len(kpts1_cv) and 
                                0 <= m.trainIdx < len(kpts2_cv)]
            else:
                # Convert numpy array matches to cv2.DMatch objects
                good_matches = []
                for i, m in enumerate(matches):
                    if m >= 0 and i < len(kpts1_cv) and m < len(kpts2_cv):
                        good_matches.append(cv2.DMatch(_imgIdx=0, _queryIdx=i, 
                                                    _trainIdx=int(m), _distance=0))
            
            if len(good_matches) == 0:
                print(f"No valid matches found to draw")
                # Return concatenated images as fallback
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
                vis[:h1, :w1] = img1
                vis[:h2, w1:w1 + w2] = img2
                return vis
            
            # Draw matches
            img_matches = cv2.drawMatches(img1, kpts1_cv,
                                        img2, kpts2_cv,
                                        good_matches, None,
                                        matchColor=color,
                                        singlePointColor=None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            return img_matches
            
        except Exception as e:
            print(f"Error in draw_matches: {str(e)}")
            # Return concatenated images as fallback
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1 + w2] = img2
            return vis
        
    def visualize_steps(self, img1: np.ndarray, img2: np.ndarray, method_name: str, 
                   method: Any) -> None:
        """
        Visualize steps of the stitching process for a single method.
        
        Args:
            img1: First image
            img2: Second image
            method_name: Name of the method
            method: Method instance
        """
        try:
            plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            
            # 1. Feature Detection and Matching
            try:
                kpts1, kpts2, matches = method.detect_and_match_features(img1, img2)
                matches_img = self.draw_matches(img1, img2, kpts1, kpts2, matches, 
                                            self.colors[method_name])
                match_text = f'Feature Matching (Found {len(matches)} matches)'
            except Exception as e:
                print(f"Error in feature matching for {method_name}: {str(e)}")
                matches_img = np.hstack([img1, img2])
                match_text = 'Feature Matching Failed'
            
            plt.subplot(gs[0, :])
            plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
            plt.title(f'{method_name}: {match_text}')
            plt.axis('off')
            
            # 2. Homography Estimation
            try:
                H = method.estimate_homography(kpts1, kpts2, matches)
                h, w = img1.shape[:2]
                warped = cv2.warpPerspective(img1, H, (w*2, h*2))
                warp_text = 'Warped Image'
            except Exception as e:
                print(f"Error in homography estimation for {method_name}: {str(e)}")
                warped = img1
                warp_text = 'Warping Failed'
            
            plt.subplot(gs[1, 0])
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title(f'{method_name}: {warp_text}')
            plt.axis('off')
            
            # 3. Final Stitching Result
            try:
                result, metadata = method.stitch_images(img1, img2)
                stitch_text = f'Final Result (Time: {metadata.get("execution_time", 0):.3f}s)'
            except Exception as e:
                print(f"Error in stitching for {method_name}: {str(e)}")
                result = np.hstack([img1, img2])
                stitch_text = 'Stitching Failed'
            
            plt.subplot(gs[1, 1])
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f'{method_name}: {stitch_text}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{method_name}_steps.png', 
                    bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing steps for {method_name}: {str(e)}")
            # Create a simple error visualization
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, f'Visualization failed for {method_name}\nError: {str(e)}',
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.savefig(self.output_dir / f'{method_name}_error.png')
            plt.close()

    def compare_methods(self, img1: np.ndarray, img2: np.ndarray) -> None:
        """
        Compare all methods on a pair of images.
        
        Args:
            img1: First image
            img2: Second image
        """
        # Create comparison figure
        n_methods = len(self.methods)
        plt.figure(figsize=(20, 5 * n_methods))
        
        for idx, (name, method) in enumerate(self.methods.items()):
            # Get results
            result, metadata = method.stitch_images(img1, img2)
            
            plt.subplot(n_methods, 1, idx + 1)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f'{name} Result (Time: {metadata["execution_time"]:.3f}s)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_feature_distribution(self, img: np.ndarray) -> None:
        """
        Visualize feature distribution for each method.
        
        Args:
            img: Input image
        """
        plt.figure(figsize=(20, 5))
        
        for idx, (name, method) in enumerate(self.methods.items()):
            # Detect features
            if hasattr(method, 'detect_features'):
                kpts = method.detect_features(img)
            else:
                # If method doesn't have separate detection, use the first output
                kpts, _, _ = method.detect_and_match_features(img, img)
            
            # Draw keypoints
            img_kpts = cv2.drawKeypoints(img, 
                                       [cv2.KeyPoint(x[0], x[1], 1) for x in kpts], 
                                       None,
                                       color=self.colors[name],
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            plt.subplot(1, len(self.methods), idx + 1)
            plt.imshow(cv2.cvtColor(img_kpts, cv2.COLOR_BGR2RGB))
            plt.title(f'{name}: {len(kpts)} keypoints')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distribution.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    import random
    import argparse
    from utils.data_loader import HPatchesDataset
    from methods.traditional.sift import SIFTStitching, default_config as sift_config
    from methods.traditional.orb import ORBStitching, default_config as orb_config
    from methods.traditional.akaze import AKAZEStitching, default_config as akaze_config
    from methods.deep_learning.superpoint import SuperPointStitching, default_config as superpoint_config
    
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-examples', type=int, default=3,
                      help='Number of random viewpoint examples to process')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Initialize dataset
    dataset = HPatchesDataset("hpatches-sequences-release")
    
    # Get all viewpoint pairs and select random examples
    v_pairs = dataset.get_pairs(type='viewpoint')
    selected_pairs = random.sample(v_pairs, min(args.num_examples, len(v_pairs)))
    
    print(f"\nSelected {len(selected_pairs)} random viewpoint pairs:")
    for i, pair in enumerate(selected_pairs):
        print(f"{i+1}. {pair.sequence_name}")
    
    # Initialize methods
    methods = {
        'SIFT': SIFTStitching(sift_config),
        'ORB': ORBStitching(orb_config),
        'AKAZE': AKAZEStitching(akaze_config),
        'SuperPoint': SuperPointStitching({
            **superpoint_config,
            'weights_path': 'SuperGluePretrainedNetwork/models/weights/superglue_indoor.pth'
        })
    }
    
    # Process each selected pair
    for i, test_pair in enumerate(selected_pairs):
        # Create a subfolder for this example
        example_dir = Path(f'visualization_results/example_{i+1}_{test_pair.sequence_name}')
        example_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nProcessing example {i+1}: {test_pair.sequence_name}")
        
        # Create visualizer with the example-specific directory
        visualizer = StitchingVisualizer(methods, output_dir=str(example_dir))
        
        # Generate visualizations for this example
        for name, method in methods.items():
            try:
                print(f"  Visualizing {name}...")
                visualizer.visualize_steps(test_pair.ref_image, test_pair.target_image, 
                                         name, method)
            except Exception as e:
                print(f"  Failed to visualize {name}: {str(e)}")
        
        try:
            print("  Generating comparison visualization...")
            visualizer.compare_methods(test_pair.ref_image, test_pair.target_image)
            visualizer.visualize_feature_distribution(test_pair.ref_image)
        except Exception as e:
            print(f"  Failed to generate comparison visualizations: {str(e)}")
            
        print(f"  Results saved in: {example_dir}")
    
    print("\nVisualization complete!")