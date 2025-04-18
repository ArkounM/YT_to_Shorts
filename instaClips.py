import os
import subprocess
import math
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import whisper
import textwrap
import time
from PIL import Image, ImageDraw, ImageFont

def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file"""
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_path} -y"
    subprocess.call(command, shell=True)
    return output_path

def transcribe_audio(video_path, audio_path, keyword="Day", min_duration=45, max_duration=60):
    """Transcribe audio using Whisper and find segments with keyword"""
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    segments = []
    
    # Process each word in the transcription
    for segment in result["segments"]:
        for i, word in enumerate(segment["words"]):
            word_text = word["word"].strip().lower()
            
            # Check if the word contains our keyword (case insensitive)
            if keyword.lower() in word_text:
                # Get timestamp info
                start_time = word["start"]
                end_time = word["end"]
                
                # Calculate clip duration to be between min_duration and max_duration seconds
                # Center the clip around the keyword if possible
                half_duration = min_duration / 2
                
                # Try to center the clip around the keyword
                context_start = max(0, start_time - half_duration)
                context_end = context_start + min_duration  # Ensure minimum duration
                
                # Cap at max_duration
                if (context_end - context_start) > max_duration:
                    context_end = context_start + max_duration
                
                # Get full text of this segment
                segment_text = segment["text"]
                
                # Find all words in the timespan of our clip
                clip_words = []
                for seg in result["segments"]:
                    for w in seg["words"]:
                        if context_start <= w["start"] <= context_end:
                            clip_words.append({
                                "text": w["word"],
                                "start": w["start"] - context_start,  # Relative to clip start
                                "end": w["end"] - context_start       # Relative to clip start
                            })
                
                # Now calculate text lines and timing for captions
                # Get video properties to determine text width
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = 1080  # Instagram width
                
                # Sample text to determine character width
                sample_text = "Sample Text for Calculation"
                font_size = 60  # Default font size
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Create a temporary image to measure text size
                temp_img = Image.new('RGB', (1, 1))
                draw = ImageDraw.Draw(temp_img)
                
                # Calculate average char width
                bbox = draw.textbbox((0, 0), sample_text, font=font)
                sample_width = bbox[2] - bbox[0]
                char_width = sample_width / len(sample_text)
                
                # Calculate usable width (80% of screen width)
                usable_width = int(width * 0.8)
                chars_per_line = int(usable_width / char_width)
                
                # Create text lines based on timing and words
                text_lines = []
                current_line = ""
                line_start_time = None
                
                for word in clip_words:
                    word_text = word["text"].strip()
                    if not word_text:
                        continue
                        
                    # Start a new line if needed
                    if line_start_time is None:
                        line_start_time = word["start"]
                    
                    # Check if adding this word would exceed line length
                    test_line = current_line + " " + word_text if current_line else word_text
                    if len(test_line) > chars_per_line:
                        # Add current line to text_lines
                        if current_line:
                            text_lines.append({
                                "text": current_line,
                                "start": line_start_time,
                                "end": word["start"]
                            })
                        
                        # Start new line with current word
                        current_line = word_text
                        line_start_time = word["start"]
                    else:
                        # Add word to current line
                        current_line = test_line
                
                # Add the last line if there is one
                if current_line:
                    text_lines.append({
                        "text": current_line,
                        "start": line_start_time,
                        "end": clip_words[-1]["end"] if clip_words else context_end - context_start
                    })
                
                cap.release()
                
                segments.append({
                    "start": context_start,
                    "end": context_end,
                    "text": segment_text,
                    "keyword": keyword,
                    "keyword_start": start_time,
                    "keyword_end": end_time,
                    "words": clip_words,
                    "text_lines": text_lines,
                    "fps": fps
                })
                
    return segments

def cv2_to_pil(cv2_img):
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_rounded_rectangle(draw, bbox, radius, fill):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    # Draw four corners
    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)

def create_instagram_clip_opencv(video_path, segment, output_path, keyword="Day", captions=False):
    """Create Instagram clip using OpenCV directly"""
    print(f"Processing clip from {segment['start']:.2f} to {segment['end']:.2f}")
    
    # Instead of trying to seek with OpenCV, extract the exact video segment with FFmpeg first
    temp_video_path = f"{output_path}_segment.mp4"
    clip_duration = segment["end"] - segment["start"]
    extract_cmd = f"ffmpeg -ss {segment['start']:.6f} -i {video_path} -t {clip_duration:.6f} -c:v copy -c:a copy {temp_video_path} -y"
    
    print(f"Extracting exact segment with FFmpeg: {extract_cmd}")
    subprocess.call(extract_cmd, shell=True)
    
    # Now open the extracted segment, which will start at time 0
    cap = cv2.VideoCapture(temp_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set output dimensions (9:16 aspect ratio)
    width = 1080    
    height = 1920
    
    # Set up video writer - using H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path + "_temp.mp4", fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Failed to open video writer. Trying different codec...")
        # Fallback to MPEG-4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path + "_temp.mp4", fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Could not open video writer with either codec. Please check your OpenCV installation.")
            return None
    
    print(f"Processing {total_frames} frames at {fps} fps")
    print(f"Expected duration: {total_frames / fps:.2f} seconds")
    
    frames_processed = 0
    
    while frames_processed < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Get original dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Create a 9:16 canvas
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create blurred background
        bg_scale = height / orig_height
        bg_width = int(orig_width * bg_scale)
        
        # Resize for background
        bg_resized = cv2.resize(frame, (bg_width, height))
        
        # Handle background sizing
        if bg_width > width:
            x_offset = (bg_width - width) // 2
            bg_resized = bg_resized[:, x_offset:x_offset+width]
        else:
            x_offset = (width - bg_width) // 2
            result[:, x_offset:x_offset+bg_width] = bg_resized
            bg_resized = result.copy()
        
        # Apply blur to background
        blurred_bg = cv2.GaussianBlur(bg_resized, (151, 151), 0)
        
        # Calculate video placement - using 70% of screen height
        target_height = int(height * 0.7)
        scale_factor = target_height / orig_height
        target_width = int(orig_width * scale_factor)
        
        # Ensure video width fits
        if target_width > width * 1.0:
            target_width = int(width * 1.0)
            scale_factor = target_width / orig_width
            target_height = int(orig_height * scale_factor)
        
        # Resize original frame to target size
        orig_resized = cv2.resize(frame, (target_width, target_height))
        
        # Center the video horizontally and position at center
        x_offset = (width - target_width) // 2
        y_offset = int(height - target_height) // 2
        
        # Create final result with blurred background
        result = blurred_bg.copy()
        result[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = orig_resized
        
        # Only add captions if enabled
        if captions:
            # Convert CV2 image to PIL for text rendering
            pil_img = cv2_to_pil(result)
            draw = ImageDraw.Draw(pil_img)
            
            try:
                # Larger font size for better readability
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Get current time relative to clip start
            current_time = (frames_processed / fps)
            
            # Find the text line for the current time
            current_line = None
            for line in segment["text_lines"]:
                if line["start"] <= current_time <= line["end"]:
                    current_line = line["text"]
                    break
            
            # If no exact match, use the last line that started before current time
            if not current_line and segment["text_lines"]:
                candidates = [line for line in segment["text_lines"] if line["start"] <= current_time]
                if candidates:
                    current_line = max(candidates, key=lambda x: x["start"])["text"]
            
            if current_line:  # Only proceed if we have text to display
                # Calculate text size
                bbox = draw.textbbox((0, 0), current_line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position text below video
                text_x = (width - text_width) // 2
                text_y = y_offset + target_height + 60  # Increased spacing
                
                # Draw rounded rectangle background
                padding = 40
                corner_radius = 30  # Adjust for rounder corners
                bg_bbox = (
                    text_x - padding,
                    text_y - padding,
                    text_x + text_width + padding,
                    text_y + text_height + padding
                )
                draw_rounded_rectangle(draw, bg_bbox, corner_radius, fill=(255, 255, 255, 230))
                
                # Draw text in black
                draw.text(
                    (text_x, text_y),
                    current_line,
                    font=font,
                    fill=(0, 0, 0)  # Black text
                )
            
            # Convert back to CV2 for video writing
            result = pil_to_cv2(pil_img)

        # Write the frame
        out.write(result)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processed {frames_processed} frames")
    print(f"Expected duration: {frames_processed / fps:.2f} seconds")
    
    # Now combine our processed video with the audio from the original video segment
    command = f"ffmpeg -i {output_path}_temp.mp4 -i {temp_video_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {output_path} -y"
    print(f"Adding audio: {command}")
    subprocess.call(command, shell=True)
    
    # Check if the output file was created successfully
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Successfully created clip at {output_path}")
        # Remove temporary files
        os.remove(f"{output_path}_temp.mp4")
        os.remove(temp_video_path)
        return output_path
    else:
        print(f"Failed to create clip at {output_path}")
        return None

def main(video_path, output_dir="instagram_clips", keyword="Day", min_duration=45, max_duration=60, captions=False):
    """Main function to process video and create Instagram clips"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract audio from video
    audio_path = extract_audio(video_path)
    
    # Transcribe audio and find segments with keyword
    segments = transcribe_audio(video_path, audio_path, keyword, min_duration, max_duration)
    
    print(f"Found {len(segments)} segments with keyword '{keyword}'")
    
    # Process each segment individually with better error handling
    clip_paths = []
    for i, segment in enumerate(segments):
        print(f"\nProcessing segment {i+1}/{len(segments)}")
        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        
        try:
            clip_path = create_instagram_clip_opencv(video_path, segment, output_path, keyword, captions)
            if clip_path:
                clip_paths.append(clip_path)
                print(f"Successfully created clip {i+1}")
            else:
                print(f"Failed to create clip {i+1}")
        except Exception as e:
            print(f"Error processing segment {i+1}: {str(e)}")
    
    # Clean up temporary files
    os.remove(audio_path)
    
    print(f"\nCreated {len(clip_paths)} clips in {output_dir}")
    return clip_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Instagram clips from long form video")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default="instagram_clips", help="Directory to save output clips")
    parser.add_argument("--keyword", default="Day", help="Keyword to detect for creating clips")
    parser.add_argument("--min-duration", type=int, default=45, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum clip duration in seconds")
    parser.add_argument("--captions", action="store_true", help="Enable captions in the output clips")
    
    args = parser.parse_args()
    
    main(args.video_path, args.output_dir, args.keyword, args.min_duration, args.max_duration, args.captions)