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

def transcribe_audio(audio_path, keyword="Day", min_duration=60, max_duration=60):
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
                
                # Calculate clip duration to be between 60-60 seconds
                # Center the clip around the keyword if possible
                half_duration = min_duration / 2  # Half of minimum duration
                
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
                                "start": w["start"],
                                "end": w["end"]
                            })
                
                segments.append({
                    "start": context_start,
                    "end": context_end,
                    "text": segment_text,
                    "keyword": keyword,
                    "keyword_start": start_time,
                    "keyword_end": end_time,
                    "words": clip_words
                })
                
    return segments

def get_sentence_at_time(segment, current_time):
    """Get the current complete sentence based on time"""
    # Group words into sentences (simple approach using basic punctuation)
    sentences = []
    current_sentence = []
    
    for word in segment["words"]:
        current_sentence.append(word)
        if word["text"].strip().endswith(('.', '!', '?')):
            sentences.append(current_sentence)
            current_sentence = []
    
    # Add any remaining words as a sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    # Find which sentence we're currently in
    current_sentence = []
    for sentence in sentences:
        sentence_start = sentence[0]["start"]
        sentence_end = sentence[-1]["end"]
        if sentence_start <= current_time <= sentence_end:
            current_sentence = sentence
            break
    
    # If no current sentence found, use the last sentence before current_time
    if not current_sentence:
        for sentence in reversed(sentences):
            if sentence[-1]["end"] <= current_time:
                current_sentence = sentence
                break
        if not current_sentence and sentences:
            current_sentence = sentences[0]
    
    # Convert sentence to text
    text = " ".join(word["text"] for word in current_sentence)
    return text.strip()

def cv2_to_pil(cv2_img):
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_instagram_clip_opencv(video_path, segment, output_path, keyword="Day", captions=False):
    """Create Instagram clip using OpenCV directly"""
    print(f"Processing clip from {segment['start']:.2f} to {segment['end']:.2f}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set output dimensions (9:16 aspect ratio)
    width = 1080    
    height = 1920
    
    # Calculate start and end frames
    start_frame = int(segment["start"] * fps)
    end_frame = int(segment["end"] * fps)
    
    # Ensure we have enough frames for the desired duration
    duration_frames = end_frame - start_frame
    min_frames = int(60 * fps)  # 60 seconds minimum
    if duration_frames < min_frames:
        end_frame = start_frame + min_frames
        # Update segment end time for audio extraction
        segment["end"] = segment["start"] + (min_frames / fps)
    
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
    
    # OpenCV font constants
    font_to_use = cv2.FONT_HERSHEY_TRIPLEX  # This is a bolder, clearer font
    
    # Print some diagnostic info
    print(f"Processing frames {start_frame} to {end_frame}")
    print(f"Total duration: {(end_frame - start_frame) / fps:.2f} seconds")
    
    # Extract the segment from the video
    precise_start_time = segment["start"]
    cap.set(cv2.CAP_PROP_POS_MSEC, precise_start_time * 1000)
    
    # Verify we're at the correct position
    actual_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    actual_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    print(f"Requested start time: {precise_start_time:.3f}, Actual time: {actual_time:.3f}")
    
    current_frame = start_frame
    frames_processed = 0
    
    while current_frame <= end_frame and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at position {current_frame}")
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
        
        # Calculate video placement - now using 60% of screen height
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
        
        # Center the video horizontally and position at 20% from top
        x_offset = (width - target_width) // 2
        y_offset = int(height - target_height) // 2  # Center
        
        # Create final result with blurred background
        result = blurred_bg.copy()
        result[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = orig_resized
        
        # Only add captions if enabled
        if captions:
            # Convert CV2 image to PIL for text rendering
            pil_img = cv2_to_pil(result)
            draw = ImageDraw.Draw(pil_img)
            
            # Load a nicer font (you'll need to provide the path to your font file)
            try:
                font = ImageFont.truetype("arial.ttf", 60)  # Adjust size as needed
            except:
                font = ImageFont.load_default()
            
            # Get current time and caption text
            current_time = segment["start"] + (frames_processed / fps)
            caption_text = get_sentence_at_time(segment, current_time)
            
            # Wrap text
            wrapped_text = textwrap.fill(caption_text, width=40)
            
            # Calculate text size and position
            bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text below video
            text_x = (width - text_width) // 2
            text_y = y_offset + target_height + 40
            
            # Draw semi-transparent background
            padding = 40
            bg_bbox = (
                text_x - padding,
                text_y - padding,
                text_x + text_width + padding,
                text_y + text_height + padding
            )
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text(
                (text_x, text_y),
                wrapped_text,
                font=font,
                fill=(255, 255, 255)
            )
            
            # Convert back to CV2 for video writing
            result = pil_to_cv2(pil_img)

        # Write the frame
        out.write(result)
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processed {frames_processed} frames")
    print(f"Expected duration: {frames_processed / fps:.2f} seconds")
    
    # Now add the audio using FFmpeg
    clip_duration = segment["end"] - segment["start"]
    print(f"Using FFmpeg to add audio from {segment['start']:.2f} for {clip_duration:.2f} seconds")
    
    audio_segment = f"-ss {segment['start']:.6f} -t {clip_duration:.6f}"
    command = f"ffmpeg -i {output_path}_temp.mp4 -i {video_path} {audio_segment} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {output_path} -y"
    print(f"Running FFmpeg command: {command}")
    subprocess.call(command, shell=True)
    
    # Check if the output file was created successfully
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Successfully created clip at {output_path}")
        # Remove temporary file
        os.remove(f"{output_path}_temp.mp4")
        return output_path
    else:
        print(f"Failed to create clip at {output_path}")
        return None

def main(video_path, output_dir="instagram_clips", keyword="Day", min_duration=60, max_duration=60, captions=False):
    """Main function to process video and create Instagram clips"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract audio from video
    audio_path = extract_audio(video_path)
    
    # Transcribe audio and find segments with keyword
    segments = transcribe_audio(audio_path, keyword, min_duration, max_duration)
    
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
    parser.add_argument("--min-duration", type=int, default=60, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum clip duration in seconds")
    parser.add_argument("--captions", action="store_true", help="Enable captions in the output clips")
    
    args = parser.parse_args()
    
    main(args.video_path, args.output_dir, args.keyword, args.min_duration, args.max_duration, args.captions)