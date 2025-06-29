import subprocess
import os

def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file"""
    print(f"Extracting audio from: {video_path} to {output_path}")
    command = f"ffmpeg -i \"{video_path}\" -ab 160k -ac 2 -ar 44100 -vn \"{output_path}\" -y"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Audio extracted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return None
    return output_path

def get_video_duration(video_path):
    """Get the duration of the video in seconds"""
    command = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return float(output)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        return None
    except FileNotFoundError:
        print("Error: ffprobe command not found. Please ensure FFmpeg is installed and in your PATH.")
        return None

def get_video_dimensions(video_path):
    """Get the width and height of the video"""
    # This function will be more robust using ffprobe, like get_video_duration
    # For now, let's assume cv2 is available where this is called, or we can add it here.
    # We'll keep it simple and expect it to be called from scripts that already import cv2.
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} to get dimensions.")
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height
