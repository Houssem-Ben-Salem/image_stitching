import numpy as np
import cv2
import time
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import os
import gc

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StitchingEvaluator:
    """Comprehensive evaluation framework for image stitching methods."""
    
    def __init__(self, dataset, methods: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            dataset: HPatchesDataset instance
            methods: Dictionary of method names and their instances
        """
        self.dataset = dataset
        self.methods = methods
        self.results = {}
        
    def compute_corner_error(self, H_est: np.ndarray, H_gt: np.ndarray, 
                           img_shape: Tuple[int, int]) -> float:
        """Compute Mean Corner Error (MCE)."""
        h, w = img_shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Transform corners
        corners_est = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_est).reshape(-1, 2)
        corners_gt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
        
        # Compute mean corner error
        mce = np.mean(np.linalg.norm(corners_est - corners_gt, axis=1))
        return mce
        
    def compute_reprojection_error(self, H_est: np.ndarray, H_gt: np.ndarray, 
                                 keypoints: np.ndarray) -> float:
        """Compute reprojection error for keypoints."""
        if len(keypoints) == 0:
            return float('inf')
            
        # Transform keypoints
        kpts_est = cv2.perspectiveTransform(keypoints.reshape(-1, 1, 2), H_est).reshape(-1, 2)
        kpts_gt = cv2.perspectiveTransform(keypoints.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
        
        # Compute mean reprojection error
        repr_error = np.mean(np.linalg.norm(kpts_est - kpts_gt, axis=1))
        return repr_error
        
    def compute_image_quality(self, result: np.ndarray, img1: np.ndarray, 
                        img2: np.ndarray, H: np.ndarray) -> Dict[str, float]:
        """Compute image quality metrics in overlapping regions."""
        try:
            # Ensure all images have the same size
            h_result, w_result = result.shape[:2]
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Resize images to match result size if needed
            if (h1, w1) != (h_result, w_result):
                img1 = cv2.resize(img1, (w_result, h_result))
            if (h2, w2) != (h_result, w_result):
                img2 = cv2.resize(img2, (w_result, h_result))
            
            # Create masks for overlapping regions
            mask1 = np.ones((h_result, w_result), dtype=np.uint8)
            mask2 = cv2.warpPerspective(mask1, H, (w_result, h_result))
            
            # Get overlapping region
            overlap = mask1 & mask2
            overlap_pixels = np.sum(overlap)
            
            if overlap_pixels == 0:
                return {
                    'ssim': 0.0,
                    'psnr': 0.0,
                    'ghosting': float('inf')
                }
            
            # Compute warped image
            warped_img1 = cv2.warpPerspective(img1, H, (w_result, h_result))
            
            # Get overlapping regions as masked images
            # This preserves spatial structure
            overlap_img1 = np.zeros_like(warped_img1)
            overlap_img2 = np.zeros_like(img2)
            
            overlap_img1[overlap > 0] = warped_img1[overlap > 0]
            overlap_img2[overlap > 0] = img2[overlap > 0]
            
            if overlap_pixels < 49:  # Less than 7x7 pixels
                return {
                    'ssim': 0.0,
                    'psnr': 0.0,
                    'ghosting': float(np.mean(np.abs(overlap_img1 - overlap_img2)))
                }
            
            # Calculate SSIM on the full images with zero-padded non-overlapping regions
            if len(img1.shape) == 3:
                ssim_value = ssim(overlap_img1, overlap_img2, 
                                win_size=7,
                                channel_axis=2)
            else:
                ssim_value = ssim(overlap_img1, overlap_img2,
                                win_size=7)
            
            # Calculate PSNR only on overlapping regions to avoid division by zero
            mse = np.mean((warped_img1[overlap > 0] - img2[overlap > 0]) ** 2)
            psnr_value = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
            
            # Calculate ghosting only on overlapping regions
            ghosting = np.mean(np.abs(warped_img1[overlap > 0] - img2[overlap > 0]))
            
            quality = {
                'ssim': float(ssim_value),
                'psnr': float(psnr_value),
                'ghosting': float(ghosting)
            }
            
            return quality
            
        except Exception as e:
            print(f"Error in compute_image_quality: {str(e)}")
            return {
                'ssim': 0.0,
                'psnr': 0.0,
                'ghosting': float('inf')
            }
        
    def compute_repeatability(self, method, img1: np.ndarray, img2: np.ndarray, 
                            H_gt: np.ndarray) -> float:
        """Compute keypoint repeatability score."""
        # Detect keypoints in both images
        kpts1, kpts2, _ = method.detect_and_match_features(img1, img2)
        
        if len(kpts1) == 0 or len(kpts2) == 0:
            return 0.0
            
        # Transform keypoints from img1 to img2 space using ground truth homography
        kpts1_warped = cv2.perspectiveTransform(kpts1.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
        
        # For each warped keypoint, find the nearest keypoint in img2
        repeated = 0
        threshold = 3.0  # pixels
        
        for kp_warped in kpts1_warped:
            distances = np.linalg.norm(kpts2 - kp_warped, axis=1)
            if np.min(distances) < threshold:
                repeated += 1
                
        return repeated / len(kpts1)
        
    def evaluate_sequence(self, sequence_type: str, pairs: List) -> Dict[str, Dict[str, float]]:
        """Evaluate methods on a specific sequence type."""
        sequence_results = {name: {
            'mce': [],
            'reprojection_error': [],
            'inlier_ratio': [],
            'num_matches': [],
            'ssim': [],
            'psnr': [],
            'ghosting': [],
            'repeatability': [],
            'time_features': [],
            'time_homography': [],
            'time_total': [],
            'memory_usage': []
        } for name in self.methods.keys()}
        
        for pair in tqdm(pairs, desc=f'Evaluating {sequence_type}'):
            for name, method in self.methods.items():
                try:
                    # Track memory usage
                    process = psutil.Process()
                    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
                    
                    # Time the process
                    start_time = time.time()
                    
                    # Feature detection and matching
                    t0 = time.time()
                    kpts1, kpts2, matches = method.detect_and_match_features(
                        pair.ref_image, pair.target_image
                    )
                    t1 = time.time()
                    
                    # Skip if not enough matches
                    if len(matches) < method.min_matches:
                        raise ValueError(f"Not enough matches found: {len(matches)}")
                    
                    # Homography estimation
                    H_est = method.estimate_homography(kpts1, kpts2, matches)
                    t2 = time.time()
                    
                    # Resize images to same dimensions if needed
                    h1, w1 = pair.ref_image.shape[:2]
                    h2, w2 = pair.target_image.shape[:2]
                    if (h1, w1) != (h2, w2):
                        max_h = max(h1, h2)
                        max_w = max(w1, w2)
                        ref_resized = cv2.resize(pair.ref_image, (max_w, max_h))
                        target_resized = cv2.resize(pair.target_image, (max_w, max_h))
                    else:
                        ref_resized = pair.ref_image
                        target_resized = pair.target_image

                    # Full stitching
                    result, metadata = method.stitch_images(
                        ref_resized, target_resized
                    )
                    end_time = time.time()
                    
                    # Memory usage
                    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
                    mem_usage = max(0, end_mem - start_mem)  # Ensure non-negative
                    
                    # Compute metrics with error checking
                    try:
                        mce = self.compute_corner_error(H_est, pair.homography, 
                                                    pair.ref_image.shape[:2])
                        mce = float(mce) if not np.isinf(mce) else None
                    except:
                        mce = None
                        
                    try:
                        repr_error = self.compute_reprojection_error(H_est, pair.homography, kpts1)
                        repr_error = float(repr_error) if not np.isinf(repr_error) else None
                    except:
                        repr_error = None
                    
                    try:
                        quality = self.compute_image_quality(result, pair.ref_image, 
                                                        pair.target_image, H_est)
                    except:
                        quality = {'ssim': None, 'psnr': None, 'ghosting': None}
                    
                    try:
                        repeatability = self.compute_repeatability(method, pair.ref_image, 
                                                                pair.target_image, pair.homography)
                    except:
                        repeatability = None
                    
                    # Get inlier ratio from metadata (with fallback to computing it)
                    inlier_ratio = metadata.get('inlier_ratio', None)
                    if inlier_ratio is None and hasattr(method, 'get_inlier_ratio'):
                        try:
                            inlier_ratio = method.get_inlier_ratio()
                        except:
                            inlier_ratio = None
                    
                    # Store results (only store valid values)
                    results = sequence_results[name]
                    if mce is not None:
                        results['mce'].append(mce)
                    if repr_error is not None:
                        results['reprojection_error'].append(repr_error)
                    if inlier_ratio is not None:
                        results['inlier_ratio'].append(inlier_ratio)
                        
                    results['num_matches'].append(len(matches))
                    
                    if quality['ssim'] is not None:
                        results['ssim'].append(quality['ssim'])
                    if quality['psnr'] is not None:
                        results['psnr'].append(quality['psnr'])
                    if quality['ghosting'] is not None:
                        results['ghosting'].append(quality['ghosting'])
                        
                    if repeatability is not None:
                        results['repeatability'].append(repeatability)
                        
                    results['time_features'].append(max(0, t1 - t0))
                    results['time_homography'].append(max(0, t2 - t1))
                    results['time_total'].append(max(0, end_time - start_time))
                    results['memory_usage'].append(mem_usage)
                    
                except Exception as e:
                    print(f"Error evaluating {name} on sequence {pair.sequence_name}: {e}")
                    
        return sequence_results
        
    def evaluate_all(self, max_percentage=100):
        """
        Evaluate methods on a percentage of the dataset.
        
        Args:
            max_percentage: Percentage of total pairs to process (1-100)
        """
        # Create output directories
        output_dir = Path('evaluation_results')
        backup_dir = Path('evaluation_backup')
        output_dir.mkdir(exist_ok=True)
        backup_dir.mkdir(exist_ok=True)
        
        try:
            # Process viewpoint sequences
            print("\nProcessing viewpoint sequences...")
            v_pairs = self.dataset.get_pairs(type='viewpoint')
            i_pairs = self.dataset.get_pairs(type='illumination')
            
            # Calculate how many pairs to process based on max_percentage (split between types)
            v_pairs_to_process = int(len(v_pairs) * (max_percentage/2) / 100)
            i_pairs_to_process = int(len(i_pairs) * (max_percentage/2) / 100)
            
            # Make sure we process at least one pair of each type if percentage > 0
            if max_percentage > 0:
                v_pairs_to_process = max(1, v_pairs_to_process)
                i_pairs_to_process = max(1, i_pairs_to_process)
            
            # Select pairs
            v_pairs = v_pairs[:v_pairs_to_process]
            i_pairs = i_pairs[:i_pairs_to_process]
            
            # Print processing information
            print(f"\nProcessing:")
            print(f"- Viewpoint: {v_pairs_to_process}/{len(self.dataset.get_pairs(type='viewpoint'))} pairs ({max_percentage/2}%)")
            print(f"- Illumination: {i_pairs_to_process}/{len(self.dataset.get_pairs(type='illumination'))} pairs ({max_percentage/2}%)")
            
            # Process viewpoint pairs
            self.results['viewpoint'] = self.evaluate_sequence('viewpoint', v_pairs)
            
            # Save intermediate results after viewpoint processing
            try:
                self.plot_results(backup_dir / 'viewpoint_complete')
            except Exception as e:
                print(f"Warning: Could not save viewpoint backup: {e}")
            
            # Process illumination pairs
            self.results['illumination'] = self.evaluate_sequence('illumination', i_pairs)
            
            # Final results saving
            self.plot_results(output_dir)
            
            # Save detailed summary
            summary_file = output_dir / 'evaluation_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Evaluation Summary (Processing {max_percentage}% of data)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Viewpoint pairs processed: {v_pairs_to_process}/{len(v_pairs)}\n")
                f.write(f"Illumination pairs processed: {i_pairs_to_process}/{len(i_pairs)}\n\n")
                
                for seq_type in self.results.keys():
                    f.write(f"\n{seq_type.upper()} SEQUENCES:\n")
                    f.write("-" * 40 + "\n")
                    
                    for name in self.methods.keys():
                        f.write(f"\n{name}:\n")
                        results = self.results[seq_type][name]
                        
                        for metric in results.keys():
                            values = results[metric]
                            if values:  # Only write if we have values
                                f.write(f"  {metric}:\n")
                                f.write(f"    Mean: {np.mean(values):.3f}\n")
                                f.write(f"    Std:  {np.std(values):.3f}\n")
                                f.write(f"    Med:  {np.median(values):.3f}\n")
            
            print(f"\nResults saved in: {output_dir}")
            self.print_summary()
            
        except Exception as e:
            print(f"\nError during evaluation. Saving backup results...")
            try:
                self.plot_results(backup_dir / 'error_backup')
                self.print_summary()
                print(f"Backup results saved in: {backup_dir}")
            except Exception as backup_error:
                print(f"Could not save backup results: {backup_error}")
            raise e

    def plot_results(self, output_dir: Path):
        """Generate plots for all metrics."""
        # Set non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot metrics
        metrics = ['mce', 'reprojection_error', 'inlier_ratio', 'num_matches', 
                'ssim', 'psnr', 'ghosting', 'repeatability', 'time_total']
                
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Get all values for current metric to determine y-axis limits
            all_values = []
            for seq_type in self.results.keys():  # This handles both viewpoint and illumination
                for name in self.methods.keys():
                    values = self.results[seq_type][name][metric]
                    # Filter out infinities and NaNs for better plotting
                    values = [v for v in values if v != float('inf') and v != float('-inf') and not np.isnan(v)]
                    if values:  # Only extend if we have valid values
                        all_values.extend(values)
            
            # Set reasonable y-axis limits
            if all_values:
                vmin, vmax = np.percentile(all_values, [5, 95])
                plt.ylim(vmin - 0.1 * (vmax - vmin), vmax + 0.1 * (vmax - vmin))
            
            # Plot boxes
            positions = []
            labels = []
            plot_data = []
            
            for seq_type in self.results.keys():
                for name in self.methods.keys():
                    values = self.results[seq_type][name][metric]
                    # Filter out infinities and NaNs
                    values = [v for v in values if v != float('inf') and v != float('-inf') and not np.isnan(v)]
                    if values:  # Only plot if we have valid values
                        positions.append(len(positions) + 1)
                        labels.append(f'{name}\n({seq_type})')
                        plot_data.append(values)
            
            if plot_data:  # Only create plot if we have data
                plt.boxplot(plot_data, positions=positions, labels=labels)
                
                plt.title(f'{metric.replace("_", " ").title()} Distribution')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(output_dir / f'{metric}_comparison.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Also save numerical results
            results_file = output_dir / f'{metric}_numerical_results.txt'
            with open(results_file, 'w') as f:
                f.write(f"Results for {metric}:\n")
                f.write("=" * 50 + "\n\n")
                for seq_type in self.results.keys():
                    f.write(f"\n{seq_type.upper()}:\n")
                    for name in self.methods.keys():
                        values = self.results[seq_type][name][metric]
                        values = [v for v in values if v != float('inf') and v != float('-inf') and not np.isnan(v)]
                        if values:
                            f.write(f"\n{name}:\n")
                            f.write(f"  Mean: {np.mean(values):.3f}\n")
                            f.write(f"  Std:  {np.std(values):.3f}\n")
                            f.write(f"  Med:  {np.median(values):.3f}\n")
                            f.write(f"  Min:  {np.min(values):.3f}\n")
                            f.write(f"  Max:  {np.max(values):.3f}\n")
                    
    def print_summary(self):
        """Print summary statistics for all methods and sequences."""
        metrics = ['mce', 'reprojection_error', 'inlier_ratio', 'num_matches', 
                  'ssim', 'psnr', 'ghosting', 'repeatability', 'time_total', 'memory_usage']
                  
        print("\nEvaluation Summary:")
        print("=" * 80)
        
        for seq_type in ['viewpoint', 'illumination']:
            print(f"\n{seq_type.upper()} SEQUENCES:")
            print("-" * 40)
            
            for name in self.methods.keys():
                print(f"\n{name}:")
                results = self.results[seq_type][name]
                
                for metric in metrics:
                    values = results[metric]
                    print(f"  {metric}:")
                    print(f"    Mean: {np.mean(values):.3f}")
                    print(f"    Std:  {np.std(values):.3f}")
                    print(f"    Med:  {np.median(values):.3f}")
                    
        print("\nDetailed results saved in plots/")

if __name__ == '__main__':
    from utils.data_loader import HPatchesDataset
    from methods.traditional.sift import SIFTStitching, default_config as sift_config
    from methods.traditional.orb import ORBStitching, default_config as orb_config
    from methods.traditional.akaze import AKAZEStitching, default_config as akaze_config
    from methods.deep_learning.superpoint import SuperPointStitching, default_config as superpoint_config
    import argparse
    
    # Add command line argument for max percentage
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-percentage', type=int, default=100,
                      help='Maximum percentage of pairs to process (1-100)')
    args = parser.parse_args()
    
    try:
        # Initialize dataset
        dataset = HPatchesDataset("hpatches-sequences-release")
        
        # Initialize methods with their default configurations
        methods = {
            'SIFT': SIFTStitching(sift_config),
            'ORB': ORBStitching(orb_config),
            'AKAZE': AKAZEStitching(akaze_config),
            'SuperPoint': SuperPointStitching({
                **superpoint_config,
                'weights_path': 'SuperGluePretrainedNetwork/models/weights/superglue_indoor.pth'
            })
        }
        
        # Create evaluator
        evaluator = StitchingEvaluator(dataset, methods)
        
        # Run evaluation with specified percentage
        evaluator.evaluate_all(max_percentage=args.max_percentage)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Saving partial results...")
        try:
            evaluator.plot_results(Path('evaluation_backup/interrupted'))
            evaluator.print_summary()
        except:
            print("Could not save partial results.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        try:
            evaluator.plot_results(Path('evaluation_backup/error'))
            evaluator.print_summary()
        except:
            print("Could not save partial results.")
        raise