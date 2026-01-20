"""
YOLO Benchmark System
======================
A professional benchmarking tool for comparing YOLO model versions (5, 8, 11, 26)
on video inference with comprehensive metrics logging.
Only highest end X models are supported.

Author: M. Huzaifa Shahbaz (mhuzaifadev@gmail.com)
Version: 2.0.0
"""

import cv2
import time
import json
import logging
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from ultralytics import YOLO
import torch


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""
    model_version: str
    model_size: str
    video_path: str
    total_frames: int
    processed_frames: int
    total_inference_time: float
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_inference_ms: float
    min_inference_ms: float
    max_inference_ms: float
    total_detections: int
    avg_detections_per_frame: float
    avg_confidence: float
    nms_time_ms: float  # For non-YOLO26 models
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return asdict(self)


class YOLOBenchmark:
    """
    YOLO Benchmark System for evaluating different YOLO versions on video inference.
    
    Supported Models (Highest End - X variants only):
    - YOLOv5: yolov5xu.pt (Ultralytics-trained, improved performance)
    - YOLOv8: yolov8x.pt
    - YOLO11: yolo11x.pt
    - YOLO26: yolo26x.pt (end-to-end, NMS-free)
    
    Detection Modes:
    - Mode 1: Detect all objects
    - Mode 2: Detect all vehicles (bicycles, cars, motorcycles, buses, trains, trucks)
    - Mode 3: Detect only humans/persons (COCO class ID: 0)
    """
    
    MODEL_VERSIONS = {
        5: "yolov5xu",  # Use 'xu' variant for improved Ultralytics-trained model
        8: "yolov8",
        11: "yolo11",
        26: "yolo26"
    }
    
    # Only highest end models (x size) are supported
    # Note: YOLO5 uses 'xu' variant (Ultralytics-trained) instead of 'x'
    MODEL_SIZE = "x"  # Fixed to extra-large/highest end
    
    # COCO dataset class IDs for filtering
    MODE_CLASS_IDS = {
        1: None,  # All classes
        2: [1, 2, 3, 5, 6, 7],  # All vehicles: bicycle(1), car(2), motorcycle(3), bus(5), train(6), truck(7)
        3: [0]     # Person
    }
    
    MODE_NAMES = {
        1: "all_objects",
        2: "vehicles_only",
        3: "humans_only"
    }
    
    def __init__(self, 
                 output_dir: str = "benchmark_results",
                 log_level: str = "INFO"):
        """
        Initialize YOLO Benchmark System
        
        Args:
            output_dir: Directory to save benchmark results
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Initialize metrics storage
        self.results: List[BenchmarkMetrics] = []
        
        # Detect and configure device
        self.device, self.device_info = self._detect_device()
        self.workers = self._get_optimal_workers()
        
        self.logger.info("=" * 80)
        self.logger.info("YOLO BENCHMARK SYSTEM INITIALIZED")
        self.logger.info(f"Device: {self.device_info}")
        self.logger.info(f"Workers: {self.workers}")
        self.logger.info("=" * 80)
    
    def _setup_logging(self, log_level: str):
        """Configure logging system"""
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("YOLOBenchmark")
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"benchmark_{timestamp}.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _detect_device(self) -> Tuple[str, str]:
        """
        Detect the best available device (CUDA > MPS > CPU)
        
        Returns:
            Tuple of (device_string, device_info_string)
        """
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA ({device_name}, {device_count} GPU(s))"
            return device, device_info
        
        # Check for MPS (Apple Silicon GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            device_info = "MPS (Apple Silicon GPU)"
            return device, device_info
        
        # Fallback to CPU
        device = "cpu"
        cpu_count = os.cpu_count() or 4
        device_info = f"CPU ({cpu_count} cores)"
        return device, device_info
    
    def _get_optimal_workers(self) -> int:
        """
        Get optimal number of workers based on device
        
        Returns:
            Number of workers to use
        """
        if self.device == "cuda":
            # For CUDA, use fewer workers (GPU handles parallelism)
            return min(4, os.cpu_count() or 4)
        elif self.device == "mps":
            # For MPS, use moderate workers
            return min(2, os.cpu_count() or 2)
        else:
            # For CPU, use maximum workers
            return os.cpu_count() or 4
    
    def _add_overlay_info(self, frame: np.ndarray, fps: float, num_detections: int, 
                          frame_width: int, frame_height: int):
        """
        Add FPS and detections overlay to frame
        
        Args:
            frame: Frame to add overlay to
            fps: Current FPS value
            num_detections: Number of detections in current frame
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        # Font settings - 25% bigger (1.25x scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.25
        thickness = 2
        
        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        
        # Get text size for FPS
        (fps_w, fps_h), fps_baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Position in right corner with padding
        padding = 10
        fps_x = frame_width - fps_w - padding
        fps_y = fps_h + padding
        
        # Draw black background for FPS
        cv2.rectangle(frame, 
                     (fps_x - 5, fps_y - fps_h - 5), 
                     (fps_x + fps_w + 5, fps_y + fps_baseline + 5), 
                     (0, 0, 0), -1)  # Black background
        
        # Draw FPS text in white
        cv2.putText(frame, fps_text, (fps_x, fps_y), font, font_scale, 
                   (255, 255, 255), thickness)  # White text
        
        # Detections text
        detections_text = f"Total Detections per frame: {num_detections}"
        
        # Get text size for detections
        (det_w, det_h), det_baseline = cv2.getTextSize(detections_text, font, font_scale, thickness)
        
        # Position below FPS block
        det_x = fps_x  # Align with FPS
        det_y = fps_y + fps_h + det_h + 15  # Below FPS with spacing
        
        # Draw white background for detections
        cv2.rectangle(frame, 
                     (det_x - 5, det_y - det_h - 5), 
                     (det_x + det_w + 5, det_y + det_baseline + 5), 
                     (255, 255, 255), -1)  # White background
        
        # Draw detections text in dark red
        cv2.putText(frame, detections_text, (det_x, det_y), font, font_scale, 
                   (0, 0, 139), thickness)  # Dark red (BGR: 0, 0, 139)
    
    def _get_model_name(self, version: int) -> str:
        """
        Get model name from version (only highest end X models supported)
        
        Args:
            version: YOLO version (5, 8, 11, 26)
        
        Returns:
            Model name string (e.g., 'yolov5xu.pt', 'yolo26x.pt')
        """
        if version not in self.MODEL_VERSIONS:
            raise ValueError(f"Unsupported YOLO version: {version}. "
                           f"Supported versions: {list(self.MODEL_VERSIONS.keys())}")
        
        model_prefix = self.MODEL_VERSIONS[version]
        
        # YOLO5 uses 'xu' variant (Ultralytics-trained), others use 'x'
        if version == 5:
            # yolov5xu already includes the 'u', just add '.pt'
            return f"{model_prefix}.pt"
        else:
            # For other versions, append 'x' size
            return f"{model_prefix}{self.MODEL_SIZE}.pt"
    
    def _load_model(self, model_name: str) -> YOLO:
        """
        Load YOLO model with optimal device configuration
        
        Args:
            model_name: Name of the model file
        
        Returns:
            Loaded YOLO model configured for optimal device
        """
        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"Using device: {self.device_info}")
        
        try:
            # Configure PyTorch for optimal performance before loading model
            if self.device == "cuda":
                # Enable TensorFloat-32 for faster computation on Ampere+ GPUs
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("✓ CUDA optimizations enabled (TF32, cuDNN benchmark)")
            elif self.device == "mps":
                self.logger.info("✓ MPS device configured")
            else:
                # For CPU, set number of threads
                torch.set_num_threads(self.workers)
                self.logger.info(f"✓ CPU configured with {self.workers} threads")
            
            # Load model - Ultralytics YOLO will use the device automatically
            # The device is set via environment or auto-detected
            model = YOLO(model_name)
            
            self.logger.info(f"✓ Model loaded successfully: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"✗ Failed to load model {model_name}: {str(e)}")
            raise
    
    def _get_video_info(self, video_path: str) -> Tuple[int, float, int, int]:
        """
        Extract video information
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (total_frames, fps, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return total_frames, fps, width, height
    
    def benchmark_video(self,
                       video_path: str,
                       version: int,
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45,
                       max_frames: Optional[int] = None,
                       visualize: bool = False,
                       save_video: bool = False,
                       mode: int = 1) -> BenchmarkMetrics:
        """
        Benchmark YOLO model on video
        
        Args:
            video_path: Path to input video
            version: YOLO version (5, 8, 11, 26) - only highest end X models
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS (ignored for YOLO26)
            max_frames: Maximum frames to process (None = all frames)
            visualize: Whether to display results during inference
            save_video: Whether to save annotated output video
            mode: Detection mode (1=all objects, 2=cars only, 3=humans only)
        
        Returns:
            BenchmarkMetrics object containing all metrics
        """
        # Validate mode
        if mode not in self.MODE_CLASS_IDS:
            raise ValueError(f"Invalid mode: {mode}. Supported modes: {list(self.MODE_CLASS_IDS.keys())}")
        
        # Get model name (only X size supported)
        model_name = self._get_model_name(version)
        
        # Get class filter for mode
        class_ids = self.MODE_CLASS_IDS[mode]
        mode_name = self.MODE_NAMES[mode]
        
        # Log benchmark start
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"BENCHMARK START: {model_name}")
        self.logger.info(f"Video: {video_path}")
        self.logger.info(f"Detection Mode: {mode} ({mode_name})")
        if class_ids:
            self.logger.info(f"Filtering classes: {class_ids}")
        else:
            self.logger.info("Detecting all classes")
        self.logger.info("=" * 80)
        
        # Load model
        model = self._load_model(model_name)
        
        # Get video info
        total_frames, video_fps, width, height = self._get_video_info(video_path)
        self.logger.info(f"Video Info: {total_frames} frames @ {video_fps:.2f} FPS "
                        f"({width}x{height})")
        
        if max_frames:
            process_frames = min(max_frames, total_frames)
            self.logger.info(f"Processing first {process_frames} frames")
        else:
            process_frames = total_frames
        
        # Setup video writer if saving
        video_writer = None
        output_video_path = None
        if save_video:
            # Generate output video filename
            video_path_obj = Path(video_path)
            base_name = video_path_obj.stem  # filename without extension
            
            # Sanitize filename: remove/replace problematic characters
            base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)  # Replace invalid chars
            base_name = base_name[:100]  # Limit length to avoid filesystem issues
            
            output_video_path = self.output_dir / f"{base_name}_yolo_{version}_x_mode{mode}_{mode_name}.mp4"
            
            # Create video writer with better codec support
            # Try 'mp4v' first, fallback to 'XVID' if needed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                video_fps,
                (width, height)
            )
            
            # Verify video writer was created successfully
            if not video_writer.isOpened():
                self.logger.warning(f"Failed to initialize video writer with 'mp4v', trying 'XVID'...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    video_fps,
                    (width, height)
                )
                
                if not video_writer.isOpened():
                    self.logger.error(f"Failed to initialize video writer. Video saving disabled.")
                    video_writer = None
                    save_video = False
                else:
                    self.logger.info(f"✓ Video writer initialized with XVID codec")
            
            if video_writer and video_writer.isOpened():
                self.logger.info(f"Saving output video to: {output_video_path}")
        
        # Initialize metrics tracking
        inference_times = []
        fps_values = []
        detection_counts = []
        confidence_scores = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        try:
            while cap.isOpened() and frame_count < process_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Record start time
                start_time = time.time()
                
                # Run inference with class filtering based on mode
                inference_kwargs = {
                    'conf': conf_threshold,
                    'iou': iou_threshold if version != 26 else 0.0,
                    'verbose': False,
                    'device': self.device  # Explicitly set device for inference
                }
                
                # Add class filtering if mode is not "all objects"
                if class_ids is not None:
                    inference_kwargs['classes'] = class_ids
                
                results = model(frame, **inference_kwargs)
                
                # Record end time
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Calculate FPS
                fps = 1000 / inference_time if inference_time > 0 else 0
                
                # Extract detection info
                detections = results[0].boxes
                num_detections = len(detections)
                
                if num_detections > 0:
                    confidences = detections.conf.cpu().numpy()
                    confidence_scores.extend(confidences.tolist())
                
                # Store metrics
                inference_times.append(inference_time)
                fps_values.append(fps)
                detection_counts.append(num_detections)
                
                # Get annotated frame for visualization/saving
                annotated_frame = results[0].plot()
                
                # Ensure frame dimensions match video writer
                if save_video and video_writer is not None:
                    # Resize if dimensions don't match (shouldn't happen, but safety check)
                    if annotated_frame.shape[:2] != (height, width):
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Add FPS and detections overlay in right corner
                self._add_overlay_info(annotated_frame, fps, num_detections, width, height)
                
                # Save frame to video if enabled
                if save_video and video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Visualization
                if visualize:
                    cv2.imshow(f"Benchmark: {model_name}", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Benchmark interrupted by user")
                        break
                
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    avg_fps_so_far = np.mean(fps_values)
                    self.logger.info(f"Processed {frame_count}/{process_frames} frames "
                                   f"(Avg FPS: {avg_fps_so_far:.2f})")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                if output_video_path:
                    self.logger.info(f"✓ Output video saved: {output_video_path}")
            if visualize:
                cv2.destroyAllWindows()
        
        # Calculate final metrics
        metrics = BenchmarkMetrics(
            model_version=f"YOLO{version}",
            model_size=self.MODEL_SIZE,
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=frame_count,
            total_inference_time=sum(inference_times) / 1000,  # Convert to seconds
            avg_fps=np.mean(fps_values) if fps_values else 0,
            min_fps=np.min(fps_values) if fps_values else 0,
            max_fps=np.max(fps_values) if fps_values else 0,
            avg_inference_ms=np.mean(inference_times) if inference_times else 0,
            min_inference_ms=np.min(inference_times) if inference_times else 0,
            max_inference_ms=np.max(inference_times) if inference_times else 0,
            total_detections=sum(detection_counts),
            avg_detections_per_frame=np.mean(detection_counts) if detection_counts else 0,
            avg_confidence=np.mean(confidence_scores) if confidence_scores else 0,
            nms_time_ms=0.0 if version == 26 else np.mean(inference_times) * 0.1,  # Estimate
            timestamp=datetime.now().isoformat()
        )
        
        # Store results
        self.results.append(metrics)
        
        # Log summary
        self._log_metrics_summary(metrics)
        
        return metrics
    
    def _log_metrics_summary(self, metrics: BenchmarkMetrics):
        """Log benchmark metrics summary"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("BENCHMARK RESULTS SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {metrics.model_version}{metrics.model_size}")
        self.logger.info(f"Processed: {metrics.processed_frames}/{metrics.total_frames} frames")
        self.logger.info("")
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  Average FPS: {metrics.avg_fps:.2f}")
        self.logger.info(f"  Min FPS: {metrics.min_fps:.2f}")
        self.logger.info(f"  Max FPS: {metrics.max_fps:.2f}")
        self.logger.info(f"  Avg Inference Time: {metrics.avg_inference_ms:.2f} ms")
        self.logger.info(f"  Min Inference Time: {metrics.min_inference_ms:.2f} ms")
        self.logger.info(f"  Max Inference Time: {metrics.max_inference_ms:.2f} ms")
        self.logger.info("")
        self.logger.info("Detection Metrics:")
        self.logger.info(f"  Total Detections: {metrics.total_detections}")
        self.logger.info(f"  Avg Detections/Frame: {metrics.avg_detections_per_frame:.2f}")
        self.logger.info(f"  Avg Confidence: {metrics.avg_confidence:.3f}")
        
        if metrics.model_version != "YOLO26":
            self.logger.info(f"  Est. NMS Time: {metrics.nms_time_ms:.2f} ms")
        else:
            self.logger.info("  NMS: Not Applicable (End-to-End)")
        
        self.logger.info("=" * 80)
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save benchmark results to JSON file
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to dict
        results_dict = {
            "benchmark_info": {
                "total_runs": len(self.results),
                "timestamp": datetime.now().isoformat()
            },
            "results": [m.to_dict() for m in self.results]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
    
    def compare_models(self, 
                      video_path: str,
                      versions: List[int],
                      **kwargs) -> List[BenchmarkMetrics]:
        """
        Compare multiple YOLO models on the same video (only highest end X models)
        
        Args:
            video_path: Path to input video
            versions: List of YOLO versions to compare (5, 8, 11, 26)
            **kwargs: Additional arguments for benchmark_video
        
        Returns:
            List of BenchmarkMetrics for all models
        """
        # Validate all versions are supported
        for version in versions:
            if version not in self.MODEL_VERSIONS:
                raise ValueError(f"Unsupported YOLO version: {version}. "
                               f"Supported: {list(self.MODEL_VERSIONS.keys())}")
        
        results = []
        
        for version in versions:
            self.logger.info("")
            self.logger.info("*" * 80)
            self.logger.info(f"COMPARING: YOLO{version}x (highest end model)")
            self.logger.info("*" * 80)
            
            metrics = self.benchmark_video(video_path, version, **kwargs)
            results.append(metrics)
        
        # Log comparison
        self._log_comparison(results)
        
        return results
    
    def _log_comparison(self, results: List[BenchmarkMetrics]):
        """Log comparison of multiple benchmark results"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("=" * 80)
        
        for i, metrics in enumerate(results, 1):
            self.logger.info(f"\n{i}. {metrics.model_version}{metrics.model_size}:")
            self.logger.info(f"   Avg FPS: {metrics.avg_fps:.2f}")
            self.logger.info(f"   Avg Inference: {metrics.avg_inference_ms:.2f} ms")
            self.logger.info(f"   Avg Detections: {metrics.avg_detections_per_frame:.2f}")
            self.logger.info(f"   Avg Confidence: {metrics.avg_confidence:.3f}")
        
        # Find best performer
        best_fps = max(results, key=lambda x: x.avg_fps)
        best_inference = min(results, key=lambda x: x.avg_inference_ms)
        
        self.logger.info("")
        self.logger.info("Best Performers:")
        self.logger.info(f"  Highest FPS: {best_fps.model_version}{best_fps.model_size} "
                        f"({best_fps.avg_fps:.2f} FPS)")
        self.logger.info(f"  Fastest Inference: {best_inference.model_version}"
                        f"{best_inference.model_size} "
                        f"({best_inference.avg_inference_ms:.2f} ms)")
        self.logger.info("=" * 80)
