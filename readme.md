# Regions Counting Using YOLOv8 (Inference on Video)

- Region counting is a method employed to tally the objects within a specified area, allowing for more sophisticated analyses when multiple regions are considered. These regions can be adjusted interactively using a Left Mouse Click, and the counting process occurs in real time.
- Regions can be adjusted to suit the user's preferences and requirements.

<div>
<p align="center">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/5ab3bbd7-fd12-4849-928e-5f294d6c3fcf" width="45%" alt="YOLOv8 region counting visual 1">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/e7c1aea7-474d-4d78-8d48-b50854ffe1ca" width="45%" alt="YOLOv8 region counting visual 2">
</p>
</div>

## Table of Contents

- [Step 1: Install the Required Libraries](#step-1-install-the-required-libraries)
- [Step 2: Run the Region Counting Using Ultralytics YOLOv8](#step-2-run-the-region-counting-using-ultralytics-yolov8)
- [Usage Options](#usage-options)
- [FAQ](#faq)

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics
```

## Step 2: Run the Region Counting Using Ultralytics YOLOv8

Here are the basic commands for running the inference:

### Note

After the video begins playing, you can freely move the region anywhere within the video by simply clicking and dragging using the left mouse button.

```bash
# If you want to save results
python region_counter.py --source "path/to/video.mp4" --save-img --view-img

# If you want to run model on CPU
python region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# If you want to change model file
python region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# If you want to detect specific class (first class and third class)
python region_counter.py --source "path/to/video.mp4" --classes 0 2 --weights "path/to/model.pt"

# If you don't want to save results
python region_counter.py --source "path/to/video.mp4" --view-img
```

## Usage Options

- `--source`: Specifies the path to the video file you want to run inference on.
- `--device`: Specifies the device `cpu` or `0`
- `--save-img`: Flag to save the detection results as images.
- `--weights`: Specifies a different YOLOv8 model file (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`).
- `--classes`: Specifies the class to be detected
- `--line-thickness`: Specifies the bounding box thickness
- `--region-thickness`: Specifies the region boxes thickness
- `--track-thickness`: Specifies the track line thickness

## FAQ

**1. How Can I Troubleshoot Issues?**

To gain more insights during inference, you can include the `--debug` flag in your command:

```bash
python yolov8_region_counter.py --source "path to video file" --debug
```

**2. Can I Employ Other YOLO Versions?**

Certainly, you have the flexibility to specify different YOLO model weights using the `--weights` option.

**3. Where Can I Access Additional Information?**

For a comprehensive guide on using YOLOv8 with Object Tracking, please refer to [Multi-Object Tracking with Ultralytics YOLO](https://docs.ultralytics.com/modes/track/).
