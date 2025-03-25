#!/usr/bin/env python3
"""
Video Depth Processor using Apple's Depth Pro.

This script processes video files and generates depth maps for each frame
using Apple's Depth Pro model.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Check if depth_pro is installed
try:
    import depth_pro
except ImportError:
    print("Error: depth_pro package not found.")
    print("Please install Apple's Depth Pro model first:")
    print("1. Clone the repository: git clone https://github.com/apple/ml-depth-pro")
    print("2. Install it: cd ml-depth-pro && pip install -e .")
    print("3. Download the models: source get_pretrained_models.sh")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the appropriate torch device based on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frame_interval: Extract every Nth frame
        
    Returns:
        Number of frames extracted
    """
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video has {frame_count} frames at {fps} FPS")
    logger.info(f"Extracting frames to {output_folder}")
    
    count = 0
    frame_number = 0
    
    with tqdm(total=frame_count // frame_interval) as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_number % frame_interval == 0:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save the frame as JPG
                frame_filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
                Image.fromarray(frame_rgb).save(frame_filename)
                
                count += 1
                pbar.update(1)
            
            frame_number += 1
    
    video.release()
    logger.info(f"Extracted {count} frames")
    return count


def process_depth_maps(frames_folder, output_folder, batch_size=1):
    """
    Process frames to create depth maps using Apple's Depth Pro.
    
    Args:
        frames_folder: Folder containing input frames
        output_folder: Folder to save depth maps
        batch_size: Number of frames to process in each batch
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "colored"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "raw"), exist_ok=True)
    
    # Load model
    logger.info("Loading Depth Pro model...")
    device = get_torch_device()
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
    )
    model.eval()
    logger.info(f"Model loaded on {device}")
    
    # Get all frame paths
    frame_paths = sorted(Path(frames_folder).glob("*.jpg"))
    total_frames = len(frame_paths)
    
    if total_frames == 0:
        logger.error("No frames found in the input folder")
        return
    
    logger.info(f"Processing {total_frames} frames for depth estimation")
    
    for i, frame_path in enumerate(tqdm(frame_paths)):
        try:
            # Load and preprocess the image
            image, _, f_px = depth_pro.load_rgb(frame_path)
            image_tensor = transform(image)
            
            # Run depth prediction
            with torch.no_grad():
                prediction = model.infer(image_tensor, f_px=f_px)
            
            # Extract depth map
            depth = prediction["depth"].detach().cpu().numpy().squeeze()
            
            # Save raw depth as NPZ
            raw_output_path = os.path.join(output_folder, "raw", f"{frame_path.stem}.npz")
            np.savez_compressed(raw_output_path, depth=depth)
            
            # Create colored visualization of the depth
            inverse_depth = 1 / depth
            max_invdepth = min(inverse_depth.max(), 1 / 0.1)  # Clip to 10cm
            min_invdepth = max(1 / 250, inverse_depth.min())  # Clip to 250m
            
            inverse_depth_normalized = (inverse_depth - min_invdepth) / (
                max_invdepth - min_invdepth
            )
            
            # Use matplotlib colormap for visualization
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("turbo")
            colored_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
            
            # Save colored depth visualization
            colored_output_path = os.path.join(output_folder, "colored", f"{frame_path.stem}.jpg")
            Image.fromarray(colored_depth).save(colored_output_path, format="JPEG", quality=90)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {str(e)}")
    
    logger.info(f"Depth processing complete. Results saved to {output_folder}")


def main():
    """Main function to process video and generate depth maps."""
    parser = argparse.ArgumentParser(
        description="Process video files to generate depth maps using Apple's Depth Pro"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input video file",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="depth_output",
        help="Output directory for depth maps",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=1,
        help="Process every Nth frame (default: 1, process all frames)",
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="Keep extracted frames after processing",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for processing (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If no input is provided, prompt user
    if not args.input:
        args.input = input("Enter path to video file: ")
    
    video_path = Path(args.input)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    output_dir = Path(args.output)
    
    # Create temporary directory for frames
    if args.keep_frames:
        frames_dir = output_dir / "frames"
    else:
        temp_dir = tempfile.TemporaryDirectory()
        frames_dir = Path(temp_dir.name)
    
    try:
        # Extract frames from video
        extract_frames(
            video_path=video_path, 
            output_folder=frames_dir, 
            frame_interval=args.frame_interval
        )
        
        # Process depth maps
        process_depth_maps(
            frames_folder=frames_dir,
            output_folder=output_dir,
            batch_size=args.batch_size
        )
        
        logger.info("Video depth processing complete!")
        logger.info(f"Depth maps saved to: {output_dir}")
        
        if args.keep_frames:
            logger.info(f"Extracted frames saved to: {frames_dir}")
    
    finally:
        # Clean up temporary directory if needed
        if not args.keep_frames:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()