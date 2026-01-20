#!/usr/bin/env python3
"""
YouTube Video Downloader with Optional Trimming
Requires: yt-dlp and ffmpeg
Install: pip install yt-dlp
"""

import subprocess
import sys
import argparse
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed. Install it with: pip install yt-dlp")
        sys.exit(1)
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed. Please install ffmpeg from https://ffmpeg.org/")
        sys.exit(1)

def download_video(url, output_path='downloads', start_time=None, end_time=None):
    """
    Download YouTube video in high quality with optional trimming.
    
    Args:
        url: YouTube video URL
        output_path: Directory to save the video
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Base command for high quality download
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', f'{output_path}/%(title)s.%(ext)s'
    ]
    
    # Add trimming with ffmpeg if timestamps provided
    if start_time is not None or end_time is not None:
        ffmpeg_args = []
        
        if start_time is not None:
            ffmpeg_args.extend(['-ss', str(start_time)])
        
        if end_time is not None:
            if start_time is not None:
                duration = end_time - start_time
                ffmpeg_args.extend(['-t', str(duration)])
            else:
                ffmpeg_args.extend(['-to', str(end_time)])
        
        # Add codec copy for fast trimming
        ffmpeg_args.extend(['-c', 'copy'])
        
        cmd.extend(['--postprocessor-args', f'ffmpeg:{" ".join(ffmpeg_args)}'])
        print(f"Downloading and trimming video (start: {start_time}s, end: {end_time}s)...")
    else:
        print("Downloading full video in high quality...")
    
    cmd.append(url)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Download completed successfully!")
        print(f"✓ Saved to: {os.path.abspath(output_path)}/")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading video: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube videos in high quality with optional trimming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full video
  python %(prog)s https://www.youtube.com/watch?v=VIDEO_ID
  
  # Download video trimmed from 30s to 90s
  python %(prog)s https://www.youtube.com/watch?v=VIDEO_ID -s 30 -e 90
  
  # Download from 1 minute to end
  python %(prog)s https://www.youtube.com/watch?v=VIDEO_ID -s 60
  
  # Download first 2 minutes only
  python %(prog)s https://www.youtube.com/watch?v=VIDEO_ID -e 120
  
  # Specify custom output directory
  python %(prog)s https://www.youtube.com/watch?v=VIDEO_ID -o my_videos
        """
    )
    
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('-s', '--start', type=float, metavar='SECONDS',
                        help='Start time in seconds')
    parser.add_argument('-e', '--end', type=float, metavar='SECONDS',
                        help='End time in seconds')
    parser.add_argument('-o', '--output', default='downloads', metavar='PATH',
                        help='Output directory (default: downloads)')
    
    args = parser.parse_args()
    
    # Validate timestamps
    if args.start is not None and args.start < 0:
        print("Error: Start time cannot be negative")
        sys.exit(1)
    
    if args.end is not None and args.end < 0:
        print("Error: End time cannot be negative")
        sys.exit(1)
    
    if args.start is not None and args.end is not None and args.start >= args.end:
        print("Error: Start time must be less than end time")
        sys.exit(1)
    
    print("Checking dependencies...")
    check_dependencies()
    
    download_video(args.url, args.output, args.start, args.end)

if __name__ == '__main__':
    main()