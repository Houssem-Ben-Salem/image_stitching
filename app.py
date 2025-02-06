import gradio as gr
import cv2
import os 
import numpy as np
from methods.traditional.sift import SIFTStitching, default_config as sift_config
from methods.traditional.orb import ORBStitching, default_config as orb_config
from methods.traditional.akaze import AKAZEStitching, default_config as akaze_config
from methods.deep_learning.superpoint import SuperPointStitching, default_config as superpoint_config

def parse_evaluation_summary(file_path='evaluation_results/evaluation_summary.txt'):
    """Parse the evaluation summary file and extract metrics for each method."""
    try:
        # Try multiple possible paths
        possible_paths = [
            'evaluation_results/evaluation_summary.txt',
            'evaluation_results_latest/evaluation_summary.txt',
            'plots/evaluation_summary.txt',
            'evaluation_summary.txt'
        ]
        
        file_content = None
        used_path = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    file_content = f.read()
                used_path = path
                break
                
        if not file_content:
            print("No evaluation summary file found")
            return {}

        print(f"Reading from: {used_path}")
        
        # Initialize data structure
        method_data = {
            'SIFT': {'viewpoint': {}, 'illumination': {}},
            'ORB': {'viewpoint': {}, 'illumination': {}},
            'AKAZE': {'viewpoint': {}, 'illumination': {}},
            'SuperPoint': {'viewpoint': {}, 'illumination': {}}
        }
        
        # Track parsing state
        current_method = None
        current_sequence = None
        current_metric = None
        
        lines = file_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and separators
            if not line or line.startswith('====') or line.startswith('----'):
                i += 1
                continue
            
            # Detect sequence type
            if 'VIEWPOINT SEQUENCES:' in line:
                current_sequence = 'viewpoint'
                print(f"\nSwitched to {current_sequence} sequence")
                i += 1
                continue
            elif 'ILLUMINATION SEQUENCES:' in line:
                current_sequence = 'illumination'
                print(f"\nSwitched to {current_sequence} sequence")
                i += 1
                continue
            
            # Detect method
            if line in ['SIFT:', 'ORB:', 'AKAZE:', 'SuperPoint:']:
                current_method = line.rstrip(':')
                print(f"\nProcessing method: {current_method}")
                i += 1
                continue
            
            # Detect metric
            if current_method and current_sequence:
                if line.endswith(':') and not line.startswith('    '):
                    current_metric = line.rstrip(':')
                    # Look ahead for Mean value
                    for j in range(1, 4):  # Look at next few lines
                        if i + j < len(lines):
                            next_line = lines[i + j].strip()
                            if next_line.startswith('Mean:'):
                                try:
                                    value = float(next_line.split(':')[1].strip())
                                    method_data[current_method][current_sequence][current_metric] = value
                                    print(f"Added {current_metric} = {value} for {current_method}/{current_sequence}")
                                except (ValueError, IndexError) as e:
                                    print(f"Error parsing value: {e}")
                                break
            i += 1

        return method_data
        
    except Exception as e:
        print(f"Error parsing evaluation summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def format_metric(value, decimal_places=3):
    """Helper function to safely format metric values."""
    try:
        if isinstance(value, (int, float)):
            if decimal_places == 0:
                return f"{value:.0f}"
            return f"{value:.{decimal_places}f}"
        return str(value)
    except:
        return 'N/A'

def update_method_info(method_name):
    """Update method information display with current evaluation results."""
    try:
        print(f"\nUpdating info for method: {method_name}")
        method_info = parse_evaluation_summary()
        
        if not method_info or method_name not in method_info:
            print(f"No data available for {method_name}")
            return f"No evaluation data available for {method_name}."
            
        info = method_info[method_name]
        vp = info.get('viewpoint', {})
        il = info.get('illumination', {})
        
        print(f"Viewpoint metrics found: {list(vp.keys())}")
        print(f"Illumination metrics found: {list(il.keys())}")
        
        display_text = f"""【 {method_name} Performance Statistics 】

📊 Viewpoint Performance:
- MCE: {format_metric(vp.get('mce', 'N/A'))} (lower is better)
- Matches: {format_metric(vp.get('num_matches', 'N/A'), 0)}
- SSIM: {format_metric(vp.get('ssim', 'N/A'))}
- PSNR: {format_metric(vp.get('psnr', 'N/A'), 1)}
- Time: {format_metric(vp.get('time_total', 'N/A'))}s
- Memory: {format_metric(vp.get('memory_usage', 0), 1)}MB

🔍 Illumination Performance:
- MCE: {format_metric(il.get('mce', 'N/A'))} (lower is better)
- Matches: {format_metric(il.get('num_matches', 'N/A'), 0)}
- SSIM: {format_metric(il.get('ssim', 'N/A'))}
- Time: {format_metric(il.get('time_total', 'N/A'))}s

💡 Key Strengths:
{get_method_strengths(method_name)}"""

        print("Generated display text:")
        print(display_text)
        return display_text
        
    except Exception as e:
        print(f"Error updating method information: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error displaying statistics for {method_name}."

def update_superglue_visibility(method):
    """Update visibility of SuperGlue model selection based on method."""
    return gr.update(visible=method == "SuperPoint")

def get_method_strengths(method_name):
    """Get the key strengths for each method."""
    strengths = {
        'SIFT': "• Robust to scale and rotation changes\n• High number of reliable matches\n• Well-tested and widely used",
        'ORB': "• Very fast processing time\n• Good illumination handling\n• Low memory usage\n• Free to use (no patents)",
        'AKAZE': "• Good balance of speed and accuracy\n• Excellent repeatability\n• Works well with local features",
        'SuperPoint': "• Best accuracy (lowest MCE)\n• Good balance of matches and speed\n• Modern deep learning approach"
    }
    return strengths.get(method_name, "")

def load_image(image_path):
    """Load and preprocess image."""
    if isinstance(image_path, np.ndarray):
        return image_path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def draw_matches_safe(img1, img2, kpts1, kpts2, matches):
    """Safely draw matches between images with proper error handling."""
    try:
        # Convert keypoints to OpenCV format
        if isinstance(kpts1, np.ndarray):
            kpts1_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts1]
        else:
            kpts1_cv = kpts1

        if isinstance(kpts2, np.ndarray):
            kpts2_cv = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kpts2]
        else:
            kpts2_cv = kpts2

        # Filter and convert matches
        good_matches = []
        if isinstance(matches[0], cv2.DMatch):
            good_matches = [m for m in matches 
                          if m.queryIdx < len(kpts1_cv) and 
                             m.trainIdx < len(kpts2_cv)]
        else:
            for i, m in enumerate(matches):
                if m >= 0 and i < len(kpts1_cv) and int(m) < len(kpts2_cv):
                    good_matches.append(cv2.DMatch(i, int(m), 0))

        if len(good_matches) == 0:
            print("No valid matches found")
            return np.hstack([img1, img2])

        # Draw matches
        return cv2.drawMatches(
            img1, kpts1_cv,
            img2, kpts2_cv,
            good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )
    except Exception as e:
        print(f"Error in draw_matches_safe: {e}")
        return np.hstack([img1, img2])

def stitch_images(image1, image2, method_name, superglue_model=None):
    """
    Stitch two images using the selected method.
    """
    try:
        # Load images
        img1 = load_image(image1)
        img2 = load_image(image2)
        
        # Initialize the selected method
        if method_name == "SIFT":
            method = SIFTStitching(sift_config)
        elif method_name == "ORB":
            method = ORBStitching(orb_config)
        elif method_name == "AKAZE":
            method = AKAZEStitching(akaze_config)
        elif method_name == "SuperPoint":
            config = superpoint_config.copy()
            config['weights_path'] = f'SuperGluePretrainedNetwork/models/weights/superglue_{superglue_model}.pth'
            method = SuperPointStitching(config)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Detect and match features
        kpts1, kpts2, matches = method.detect_and_match_features(img1, img2)
        
        # Create matches visualization
        matches_img = draw_matches_safe(img1, img2, kpts1, kpts2, matches)
        
        # Perform stitching
        result, metadata = method.stitch_images(img1, img2)
        
        # Get method statistics
        method_stats = parse_evaluation_summary().get(method_name, {})
        
        # Prepare info text with both process info and current statistics
        info_text = f"""
Process Information:
- Method: {method_name}
- Number of matches: {len(matches)}
- Execution time: {metadata['execution_time']:.2f} seconds
- Inlier ratio: {metadata.get('inlier_ratio', 'N/A')}

Current Statistics:
- Mean Corner Error (viewpoint): {method_stats.get('viewpoint', {}).get('mce', 'N/A')}
- Number of matches (avg): {method_stats.get('viewpoint', {}).get('num_matches', 'N/A')}
- Processing time (avg): {method_stats.get('viewpoint', {}).get('time_total', 'N/A')} seconds
        """
        
        return {
            matches_output: matches_img,
            result_output: result,
            process_info: info_text
        }
        
    except Exception as e:
        error_msg = f"Error during stitching: {str(e)}"
        print(error_msg)
        # Return horizontal stack of images as fallback
        fallback_img = np.hstack([img1, img2]) if 'img1' in locals() and 'img2' in locals() else None
        return {
            matches_output: fallback_img,
            result_output: fallback_img,
            process_info: error_msg
        }

# Create Gradio interface
with gr.Blocks(title="Image Stitching App") as demo:
    gr.Markdown("# Image Stitching App")
    gr.Markdown("Upload two images and select a stitching method.")
    
    with gr.Row():
        # Left column for inputs and method info
        with gr.Column(scale=2):
            # Input images
            with gr.Row():
                image1_input = gr.Image(label="First Image", type="numpy")
                image2_input = gr.Image(label="Second Image", type="numpy")
            
            with gr.Row():
                # Method selection
                method_input = gr.Dropdown(
                    choices=["SIFT", "ORB", "AKAZE", "SuperPoint"],
                    label="Stitching Method",
                    value="SIFT"
                )
                # SuperGlue model selection
                superglue_input = gr.Dropdown(
                    choices=["indoor", "outdoor"],
                    label="SuperGlue Model",
                    value="indoor",
                    visible=False
                )
            
            # Method performance statistics
            method_info = gr.Textbox(
                label="Method Performance Statistics",
                value="Select a method to see performance statistics",
                lines=12,
                interactive=False
            )
            
            # Submit button
            submit_btn = gr.Button("Stitch Images", variant="primary")
        
        # Right column for outputs
        with gr.Column(scale=2):
            matches_output = gr.Image(label="Feature Matches")
            result_output = gr.Image(label="Stitching Result")
            process_info = gr.Textbox(label="Process Information", lines=4)
    
    # Event handlers
    def on_method_change(method):
        """Handle method change event."""
        stats = update_method_info(method)
        visibility = update_superglue_visibility(method)
        return [stats, visibility]
    
    # Connect events with proper return values
    method_input.change(
        fn=on_method_change,
        inputs=[method_input],
        outputs=[method_info, superglue_input]
    )
    
    # Connect the submit button
    submit_btn.click(
        fn=stitch_images,
        inputs=[image1_input, image2_input, method_input, superglue_input],
        outputs=[matches_output, result_output, process_info]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()