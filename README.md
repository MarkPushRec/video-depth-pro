# Video Depth Pro

A Python tool for generating depth maps from videos using Apple's [Depth Pro model](https://github.com/apple/ml-depth-pro).

This script:
1. Takes a video file as input
2. Extracts frames from the video
3. Processes each frame with Apple's Depth Pro model
4. Generates depth maps for each frame in both raw and colored formats

## Requirements

- Python 3.9+
- Apple's Depth Pro model and its dependencies
- OpenCV
- PyTorch
- Additional dependencies in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MarkPushRec/video-depth-pro.git
cd video-depth-pro
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Install Apple's Depth Pro model:
```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh
cd ..
```

## Usage

Run the script with a video file:

```bash
python video_depth_processor.py -i /path/to/your/video.mp4 -o depth_output
```

### Command Line Arguments

- `-i`, `--input`: Path to input video file
- `-o`, `--output`: Output directory for depth maps (default: "depth_output")
- `--frame-interval`: Process every Nth frame (default: 1, process all frames)
- `--keep-frames`: Keep extracted frames after processing
- `--batch-size`: Batch size for processing (default: 1)
- `-v`, `--verbose`: Enable verbose logging

If no input file is specified, the script will prompt you to enter one.

## Output

The script generates two folders in the output directory:

- `raw/`: Contains raw depth data in NPZ format
- `colored/`: Contains depth maps visualized with the "turbo" colormap

If `--keep-frames` is enabled, a third folder will be created:
- `frames/`: Contains the extracted frames from the video

## Examples

Process all frames in a video:
```bash
python video_depth_processor.py -i my_video.mp4
```

Process every 5th frame and keep the extracted frames:
```bash
python video_depth_processor.py -i my_video.mp4 --frame-interval 5 --keep-frames
```

## License

This project is released under the same terms as Apple's Depth Pro model. See [Apple's license](https://github.com/apple/ml-depth-pro/blob/main/LICENSE) for details.

## Acknowledgements

- [Apple's Depth Pro model](https://github.com/apple/ml-depth-pro)
- This tool is not affiliated with or endorsed by Apple Inc.