from typing import Union, List, Optional
import logging
import torch

# Import YOLOBenchmark
from yolo_benchmark import YOLOBenchmark

# GPU utilization monitoring
# Requires: pip install nvidia-ml-py
# Note: Package name is 'nvidia-ml-py' but import is 'pynvml'
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except (ImportError, Exception):
    GPU_MONITORING_AVAILABLE = False
    pynvml = None  # type: ignore


def get_gpu_utilization() -> str:
    """
    Get GPU utilization percentage for NVIDIA GPUs
    
    Note: GPU utilization is measured as 0-100% (not above 100%).
    Low utilization (e.g., 14-29%) typically indicates:
    - CPU bottleneck (data loading/preprocessing)
    - Small batch size (processing one frame at a time)
    - Memory bandwidth limitations
    - Synchronization overhead between CPU and GPU
    
    Returns:
        String with GPU utilization percentage, or empty string if not available
    """
    if not GPU_MONITORING_AVAILABLE:
        return ""
    
    if not torch.cuda.is_available():
        return ""
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # GPU utilization is always 0-100%, representing percentage of time GPU is busy
        return f" [GPU: {utilization.gpu}%]"
    except Exception as e:
        # Silently fail - GPU monitoring is optional
        return ""


class GPUUtilizationFormatter(logging.Formatter):
    """Custom formatter that appends GPU utilization to log messages"""
    
    def format(self, record):
        # Get the original formatted message
        msg = super().format(record)
        # Append GPU utilization if available
        gpu_info = get_gpu_utilization()
        return msg + gpu_info


def setup_gpu_logging(benchmark: YOLOBenchmark):
    """
    Patch the benchmark logger to include GPU utilization in log messages
    
    Args:
        benchmark: YOLOBenchmark instance
    """
    if not GPU_MONITORING_AVAILABLE:
        print("GPU monitoring not available (install: pip install nvidia-ml-py)")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available, GPU monitoring disabled")
        return
    
    # Create custom formatter with GPU utilization
    gpu_formatter = GPUUtilizationFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Update all console handlers (not file handlers) to use GPU formatter
    # Also remove any duplicate handlers first
    console_handlers = []
    for handler in benchmark.logger.handlers[:]:  # Copy list to avoid modification during iteration
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # Check if we already have a GPU formatter handler
            if isinstance(handler.formatter, GPUUtilizationFormatter):
                continue  # Skip if already has GPU formatter
            console_handlers.append(handler)
    
    # Update console handlers
    updated = False
    for handler in console_handlers:
        handler.setFormatter(gpu_formatter)
        updated = True
    
    if updated:
        print("✓ GPU utilization monitoring enabled - GPU % will appear after each log line")
    else:
        print("⚠ Warning: No console handlers found to update with GPU formatter")


def benchmark_yolo(
    video_path: str,
    version: Union[int, List[int]] = 26,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_frames: Optional[int] = None,
    visualize: bool = False,
    save_video: bool = True,
    mode: int = 1,
    output_dir: str = "benchmark_results",
    save_results: bool = True,
    log_level: str = "INFO"
) -> Union[dict, List[dict]]:
    """
    Driver function to benchmark YOLO models on video inference.
    
    This function provides a simple interface to run comprehensive benchmarks
    on YOLO models (versions 5, 8, 11, 26) with automatic metrics collection
    and logging. Only highest end X models are supported.
    
    Parameters:
    ----------
    video_path : str
        Path to the input video file
        
    version : int or List[int], default=26
        YOLO version(s) to benchmark (only highest end X models):
        - Single version: 5, 8, 11, or 26
        - Multiple versions: [5, 8, 11, 26] for comparison
        Default uses YOLO26x (highest end model)
        
    conf_threshold : float, default=0.25
        Confidence threshold for detections (0.0 - 1.0)
        Lower = more detections, higher = more confident detections
        
    iou_threshold : float, default=0.45
        IOU threshold for NMS (ignored for YOLO26 which is NMS-free)
        
    max_frames : int or None, default=None
        Maximum number of frames to process
        - None: Process entire video
        - int: Process only first N frames (useful for quick tests)
        
    visualize : bool, default=False
        Whether to display real-time visualization during inference
        Press 'q' to stop visualization early
        
    save_video : bool, default=True
        Whether to save annotated output videos for each model
        Videos are saved with naming: video_name_yolo_{version}_x_mode{mode}_{mode_name}.mp4
        
    mode : int, default=1
        Detection mode:
        - 1: Detect all objects
        - 2: Detect all vehicles (bicycles, cars, motorcycles, buses, trains, trucks)
        - 3: Detect only humans/persons (COCO class ID: 0)
        
    output_dir : str, default="benchmark_results"
        Directory to save benchmark results, logs, and output videos
        
    save_results : bool, default=True
        Whether to save results to JSON file
        
    log_level : str, default="INFO"
        Logging verbosity: "DEBUG", "INFO", "WARNING", "ERROR"
    
    Returns:
    -------
    dict or List[dict]
        Benchmark metrics dictionary/dictionaries containing:
        - model_version: YOLO version
        - model_size: Model size variant
        - video_path: Input video path
        - total_frames: Total frames in video
        - processed_frames: Frames actually processed
        - total_inference_time: Total time spent on inference (seconds)
        - avg_fps: Average frames per second
        - min_fps, max_fps: FPS range
        - avg_inference_ms: Average inference time per frame (milliseconds)
        - min_inference_ms, max_inference_ms: Inference time range
        - total_detections: Total objects detected
        - avg_detections_per_frame: Average detections per frame
        - avg_confidence: Average detection confidence
        - nms_time_ms: Estimated NMS time (0 for YOLO26)
        - timestamp: Benchmark timestamp
    
    Examples:
    --------
    # Example 1: Quick benchmark of YOLO26x (highest end model)
    >>> results = benchmark_yolo("traffic.mp4")
    
    # Example 2: Compare all YOLO versions (highest end X models)
    >>> results = benchmark_yolo(
    ...     video_path="traffic.mp4",
    ...     version=[5, 8, 11, 26]
    ... )
    
    # Example 3: Test YOLO26x with visualization and detect all vehicles
    >>> results = benchmark_yolo(
    ...     video_path="traffic.mp4",
    ...     version=26,
    ...     mode=2,
    ...     visualize=True
    ... )
    
    # Example 4: Quick test on first 100 frames, detect only humans
    >>> results = benchmark_yolo(
    ...     video_path="traffic.mp4",
    ...     version=26,
    ...     mode=3,
    ...     max_frames=100
    ... )
    
    # Example 5: Compare all versions detecting all vehicles
    >>> results = benchmark_yolo(
    ...     video_path="traffic.mp4",
    ...     version=[5, 8, 11, 26],
    ...     mode=2
    ... )
    
    Notes:
    -----
    - Only highest end X models are supported (yolov5xu, yolov8x, yolo11x, yolo26x)
    - YOLO5 uses 'xu' variant (Ultralytics-trained) for improved performance
    - YOLO26 is end-to-end and NMS-free, making it faster on CPUs
    - Mode 1 detects all COCO classes (80 classes)
    - Mode 2 filters to detect all vehicles (bicycles, cars, motorcycles, buses, trains, trucks)
    - Mode 3 filters to detect only persons/humans (class ID: 0)
    - Results are automatically saved to JSON in output_dir
    - Logs are saved to output_dir/logs/
    """
    
    # Initialize benchmark system
    benchmark = YOLOBenchmark(output_dir=output_dir, log_level=log_level)
    
    # Setup GPU utilization monitoring for Colab
    setup_gpu_logging(benchmark)
    
    # Test GPU monitoring and inform user about status
    if GPU_MONITORING_AVAILABLE and torch.cuda.is_available():
        test_gpu = get_gpu_utilization()
        if test_gpu:
            benchmark.logger.info(f"GPU utilization monitoring enabled{test_gpu}")
        else:
            benchmark.logger.info("GPU monitoring available but unable to read GPU stats")
    elif torch.cuda.is_available():
        benchmark.logger.info("GPU utilization monitoring unavailable (install: pip install nvidia-ml-py)")
    else:
        benchmark.logger.info("CUDA not available - running on CPU")
    
    # Determine if single or multiple versions
    is_comparison = isinstance(version, list)
    
    if is_comparison:
        # Multiple model comparison
        benchmark.logger.info("=" * 80)
        benchmark.logger.info("MULTI-MODEL COMPARISON MODE")
        benchmark.logger.info(f"Comparing {len(version)} model(s): {version}")
        benchmark.logger.info("=" * 80)
        
        results = benchmark.compare_models(
            video_path=video_path,
            versions=version,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_frames=max_frames,
            visualize=visualize,
            save_video=save_video,
            mode=mode
        )
        
        # Convert to dict list
        results_dict = [r.to_dict() for r in results]
        
    else:
        # Single model benchmark
        benchmark.logger.info("=" * 80)
        benchmark.logger.info("SINGLE MODEL BENCHMARK MODE")
        benchmark.logger.info(f"Model: YOLO{version}x (highest end)")
        benchmark.logger.info("=" * 80)
        
        result = benchmark.benchmark_video(
            video_path=video_path,
            version=version,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_frames=max_frames,
            visualize=visualize,
            save_video=save_video,
            mode=mode
        )
        
        results_dict = result.to_dict()
    
    # Save results if requested
    if save_results:
        benchmark.save_results()
    
    return results_dict


def main():
    """
    Main entry point for command-line usage.
    
    Example command-line usage:
    --------------------------
    python driver.py
    
    This will run a demo benchmark on a sample video.
    Modify the parameters below for your specific use case.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLO Benchmark - Video Inference Performance Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark YOLO26x (highest end) on video
  python driver.py --video traffic.mp4
  
  # Compare multiple YOLO versions (highest end X models)
  python driver.py --video traffic.mp4 --versions 5 8 11 26
  
  # Detect all vehicles
  python driver.py --video traffic.mp4 --mode 2
  
  # Detect only humans
  python driver.py --video traffic.mp4 --mode 3
  
  # Quick test with visualization
  python driver.py --video traffic.mp4 --max-frames 100 --visualize
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--version", "--versions",
        type=int,
        nargs='+',
        default=[26],
        help="YOLO version(s) to benchmark (5, 8, 11, 26) - only highest end X models"
    )
    
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Detection mode: 1=all objects, 2=vehicles only, 3=humans only (default: 1)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all frames)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show real-time visualization"
    )
    
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Don't save annotated output videos"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/yolo-tests/benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Prepare version argument
    version = args.version if len(args.version) > 1 else args.version[0]
    
    # Run benchmark
    results = benchmark_yolo(
        video_path=args.video,
        version=version,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_frames=args.max_frames,
        visualize=args.visualize,
        save_video=not args.no_save_video,
        mode=args.mode,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        log_level=args.log_level
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Example usage when run directly
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        YOLO BENCHMARK SYSTEM                             ║
    ║                                                                          ║
    ║  Professional benchmarking tool for YOLO models (v5, v8, v11, v26)       ║
    ║  Only highest end X models supported                                     ║
    ║                                                                          ║
    ║  Usage:                                                                  ║
    ║    python yolo_benchmark_driver.py --video <path> [options]              ║
    ║                                                                          ║
    ║  For help:                                                               ║
    ║    python driver.py --help                                               ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run main function with command-line arguments
    # main()

    results = benchmark_yolo(video_path="/content/uk.mp4",version=[5, 8, 11, 26], mode=1, output_dir="/content/drive/MyDrive/yolo-tests/benchmark_results")
