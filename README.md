# YOLO Benchmark Suite

### Professional Benchmarking Tool for YOLO Models

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)

**Compare YOLOv5, YOLOv8, YOLO11, and YOLO26 performance on your hardware**

[View Benchmark Report](blog.md) | [Report Bug](https://github.com/mhuzaifadev/ultralytics_yolo_comparision/issues) | [Request Feature](https://github.com/mhuzaifadev/ultralytics_yolo_comparision/issues)

---

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
  - [Google Colab](#google-colab-easiest)
  - [Local Installation](#local-installation)
- [Usage](#usage)
- [Detection Modes](#detection-modes)
- [Output](#output)
- [Benchmark Results](#benchmark-results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Benchmark YOLOv5, YOLOv8, YOLO11, and YOLO26 side-by-side |
| **Comprehensive Metrics** | FPS, inference time, detection counts, confidence scores |
| **Video Output** | Save annotated videos with real-time metrics overlay |
| **Detection Modes** | All objects, vehicles only, or humans only |
| **Cross-Platform** | Works on Windows, macOS, Linux, and Google Colab |
| **GPU Optimized** | Automatic CUDA, MPS (Apple Silicon), and CPU support |
| **GPU Monitoring** | Real-time GPU utilization tracking (NVIDIA) |

---

## Supported Models

> **Note:** Currently, only the highest-end **X models** are supported. More model sizes coming soon!

| Model | Variant | Architecture | NMS | Status |
|-------|---------|--------------|-----|--------|
| YOLOv5 | yolov5xu | Classic + Ultralytics | Required | Supported |
| YOLOv8 | yolov8x | Anchor-free | Required | Supported |
| YOLO11 | yolo11x | Latest improvements | Required | Supported |
| YOLO26 | yolo26x | End-to-End | **NMS-Free** | Supported |

### Coming Soon

- [ ] Nano (n) variants
- [ ] Small (s) variants  
- [ ] Medium (m) variants
- [ ] Large (l) variants
- [ ] ONNX Runtime support
- [ ] TensorRT optimization

---

## Quick Start

### Google Colab (Easiest)

Run benchmarks instantly without any local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhuzaifadev/ultralytics_yolo_comparision/blob/main/colab_notebook.ipynb)

**Or manually in Colab:**

```python
# Clone the repository
!git clone https://github.com/mhuzaifadev/ultralytics_yolo_comparision.git
%cd ultralytics_yolo_comparision

# Install dependencies
!pip install -r requirements.txt
!pip install nvidia-ml-py  # For GPU monitoring

# Run benchmark
from colab_driver import benchmark_yolo

results = benchmark_yolo(
    video_path="your_video.mp4",
    version=[5, 8, 11, 26],
    mode=1  # All objects
)
```

---

### Local Installation

#### Windows

**Prerequisites:**
- Python 3.9 or higher
- NVIDIA GPU (optional, for CUDA acceleration)

**Steps:**

```powershell
# 1. Clone the repository
git clone https://github.com/mhuzaifadev/ultralytics_yolo_comparision.git
cd ultralytics_yolo_comparision

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) For NVIDIA GPU monitoring
pip install nvidia-ml-py

# 5. Run benchmark
python yolo_benchmark_driver.py --video your_video.mp4 --versions 5 8 11 26
```

---

#### macOS

**Prerequisites:**
- Python 3.9 or higher
- Apple Silicon (M1/M2/M3) recommended for MPS acceleration

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/mhuzaifadev/ultralytics_yolo_comparision.git
cd ultralytics_yolo_comparision

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run benchmark
python yolo_benchmark_driver.py --video your_video.mp4 --versions 5 8 11 26
```

> **Tip:** On Apple Silicon, the benchmark automatically uses MPS (Metal Performance Shaders) for GPU acceleration.

---

#### Linux

**Prerequisites:**
- Python 3.9 or higher
- NVIDIA GPU with CUDA (optional)

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/mhuzaifadev/ultralytics_yolo_comparision.git
cd ultralytics_yolo_comparision

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) For NVIDIA GPU monitoring
pip install nvidia-ml-py

# 5. Run benchmark
python yolo_benchmark_driver.py --video your_video.mp4 --versions 5 8 11 26
```

---

## Usage

### Basic Commands

```bash
# Benchmark single model (YOLO26x - default)
python yolo_benchmark_driver.py --video traffic.mp4

# Compare all YOLO versions
python yolo_benchmark_driver.py --video traffic.mp4 --versions 5 8 11 26

# Detect only vehicles
python yolo_benchmark_driver.py --video traffic.mp4 --versions 26 --mode 2

# Detect only humans
python yolo_benchmark_driver.py --video crowd.mp4 --versions 26 --mode 3

# Quick test (first 100 frames only)
python yolo_benchmark_driver.py --video traffic.mp4 --max-frames 100

# Show real-time visualization
python yolo_benchmark_driver.py --video traffic.mp4 --visualize
```

### All Command Line Options

```bash
python yolo_benchmark_driver.py --help
```

| Argument | Description | Default |
|----------|-------------|---------|
| --video | Path to input video file | *Required* |
| --versions | YOLO version(s): 5, 8, 11, 26 | 26 |
| --mode | Detection mode (1, 2, or 3) | 1 |
| --conf | Confidence threshold (0.0-1.0) | 0.25 |
| --iou | IOU threshold for NMS | 0.45 |
| --max-frames | Limit frames to process | All frames |
| --visualize | Show real-time preview window | Off |
| --no-save-video | Do not save annotated output videos | Save videos |
| --output-dir | Directory for results | benchmark_results |
| --log-level | Logging verbosity | INFO |

---

## Detection Modes

| Mode | Name | Description | COCO Classes |
|------|------|-------------|--------------|
| 1 | **All Objects** | Detect all 80 COCO classes | All |
| 2 | **Vehicles Only** | Cars, buses, trucks, motorcycles, bicycles, trains | 1, 2, 3, 5, 6, 7 |
| 3 | **Humans Only** | Pedestrian/person detection | 0 |

**Examples:**

```bash
# Mode 1: All objects (general benchmarking)
python yolo_benchmark_driver.py --video city.mp4 --mode 1

# Mode 2: Vehicles only (traffic analysis)
python yolo_benchmark_driver.py --video highway.mp4 --mode 2

# Mode 3: Humans only (crowd analysis)
python yolo_benchmark_driver.py --video crowd.mp4 --mode 3
```

---

## Output

After running a benchmark, you will find these files in your output directory:

```
benchmark_results/
├── logs/
│   └── benchmark_20260120_120530.log          # Detailed execution logs
│
├── benchmark_results_20260120_120530.json     # Metrics in JSON format
│
├── traffic_yolo_5_x_mode1_all_objects.mp4     # YOLOv5 annotated video
├── traffic_yolo_8_x_mode1_all_objects.mp4     # YOLOv8 annotated video
├── traffic_yolo_11_x_mode1_all_objects.mp4    # YOLO11 annotated video
└── traffic_yolo_26_x_mode1_all_objects.mp4    # YOLO26 annotated video
```

### Video Overlay Features

Each output video includes:

| Overlay | Position | Color |
|---------|----------|-------|
| **FPS (Xms)** | Top-left | White on black |
| **Total Detections** | Below FPS | Dark red on white |
| **Bounding Boxes** | On objects | Class-specific colors |

> **Note:** FPS and inference time update every half second (smoothed), while detection count updates every frame.

---

## Benchmark Results

We have conducted extensive benchmarks comparing all models. View the full analysis:

**[Read the Full Benchmark Report](blog.md)**

### Quick Performance Summary

| Platform | Winner | Avg FPS | Key Advantage |
|----------|--------|---------|---------------|
| **Apple M1 Pro** | YOLO26x | 18.2 | 86.5% GPU utilization |
| **NVIDIA T4** | YOLO26x | 33.9 | NMS-free = no CPU bottleneck |

### Key Findings

- YOLO26 is consistently fastest across all platforms
- YOLO26 achieves highest GPU utilization (86.5% on M1 Pro)
- Detection quality is maintained across models
- NMS-free architecture eliminates CPU bottleneck

---

## Troubleshooting

### CUDA Out of Memory

If you see memory errors on GPU:

```bash
# Reduce frame count for testing
python yolo_benchmark_driver.py --video video.mp4 --max-frames 500

# Or process one model at a time
python yolo_benchmark_driver.py --video video.mp4 --versions 26
```

### Model Download Fails

Models download automatically from Ultralytics. If it fails:

1. Check your internet connection
2. Download manually from [Ultralytics Releases](https://github.com/ultralytics/ultralytics/releases)
3. Place .pt files in the project root directory

### Video Codec Issues

If video saving fails:

- The tool automatically tries fallback codecs (mp4v then XVID)
- Ensure FFmpeg is installed on your system
- Use .mp4 format for best compatibility

### Apple Silicon (MPS) Issues

If you encounter MPS errors on macOS:

```bash
# Enable CPU fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1
python yolo_benchmark_driver.py --video video.mp4
```

### Duplicate Log Messages

If you see duplicate logs in Colab, this is a known issue with Jupyter notebooks. The benchmark still runs correctly.

---

## Contributing

Contributions are welcome! Here is how you can help:

1. **Fork** the repository
2. **Create** a feature branch (git checkout -b feature/AmazingFeature)
3. **Commit** your changes (git commit -m 'Add AmazingFeature')
4. **Push** to the branch (git push origin feature/AmazingFeature)
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add support for smaller model sizes (n, s, m, l)
- [ ] TensorRT export and benchmarking
- [ ] Web UI for results visualization
- [ ] Batch video processing
- [ ] Custom model support
- [ ] Real-time webcam benchmarking

---

## License

This project is licensed under the **Apache License 2.0**.

```
Copyright 2026 M. Huzaifa Shahbaz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See the [LICENSE](LICENSE) file for full details.

---

## Author

### M. Huzaifa Shahbaz

[![GitHub](https://img.shields.io/badge/GitHub-mhuzaifadev-181717?style=for-the-badge&logo=github)](https://github.com/mhuzaifadev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mhuzaifadev-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/mhuzaifadev)
[![Website](https://img.shields.io/badge/Website-mhuzaifa.com-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://mhuzaifa.com)

---

## Show Your Support

If you found this project useful, please consider:

- **Starring** this repository
- **Forking** for your own experiments
- **Sharing** with others who might benefit
- **Reporting** bugs or suggesting features

---

Made with love by [M. Huzaifa Shahbaz](https://mhuzaifa.com)
