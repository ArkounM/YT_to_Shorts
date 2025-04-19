import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import whisper
import requests
import argparse
import time
import textwrap
from typing import List, Dict, Any
import google.generativeai as genai

# Free LLM API options - we'll use Google Gemini API with a rate limit for free tier
# Alternative options include HuggingFace Inference API or other free tier services
# You can swap this implementation for any other free LLM API
class LLMClipFinder:
    """Class to handle LLM API calls for identifying interesting clips"""

    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        """Initialize with optional API key (for Google Gemini)"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model

        if not self.api_key:
            print("No Google Gemini API key found. Falling back to alternate method.")
            self.use_gemini = False
            return
            
        # Configure the Gemini API client
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.use_gemini = True
        except Exception as e:
            print(f"Failed to initialize Gemini API: {e}")
            self.use_gemini = False

    def find_interesting_moments(self, transcription_segments, min_clips=3, max_clips=10):
        """Use LLM to identify interesting moments from transcription segments"""
        
        # Format the transcription data for the LLM
        transcript_text = ""
        for i, segment in enumerate(transcription_segments):
            start_time = self._format_time(segment["start"])
            end_time = self._format_time(segment["end"])
            transcript_text += f"[{start_time} - {end_time}] {segment['text']}\n"
        
        # Create prompt for the LLM
        prompt = f"""
You are an expert video editor who finds the most compelling moments in videos.

Here's a transcript with timestamps:

{transcript_text}

Please identify {min_clips}-{max_clips} moments that would make great short clips (45-60 seconds each). Focus on:
1. Interesting statements or stories
2. Emotional moments
3. Surprising revelations or insights
4. Quotable or memorable segments
5. Self-contained moments that work well in isolation

Format your response as JSON with this structure:
{{
  "clips": [
    {{
      "start": "mm:ss",
      "end": "mm:ss",
      "reason": "brief explanation",
      "caption": "suggested caption"
    }},
    ...
  ]
}}
"""

        if self.use_gemini:
            return self._call_gemini_api(prompt)
        else:
            return self._fallback_extraction(transcription_segments)
            
    def _call_gemini_api(self, prompt):
        """Call Gemini API with proper error handling"""
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Try to parse the JSON from the response
            try:
                # Find JSON in the response if it's not pure JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                
                import json
                clip_data = json.loads(content)
                return clip_data
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response. Using manual extraction.")
                return self._manually_extract_clips(content)
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return self._fallback_extraction(self.transcription_segments)
    def _manually_extract_clips(self, content):
        """Manually extract clip information if JSON parsing fails"""
        clips = []
        
        # Try to find and extract clip information using regex
        import re
        
        # Look for patterns like "Start: 01:23" or "Start time: 01:23"
        start_times = re.findall(r'start(?:\s+time)?:\s*(\d+:\d+)', content, re.IGNORECASE)
        end_times = re.findall(r'end(?:\s+time)?:\s*(\d+:\d+)', content, re.IGNORECASE)
        
        # Extract everything between "Reason:" and the next section as the reason
        reasons = re.findall(r'reason:\s*(.*?)(?=\n\s*(?:caption|start|end|clip|\d+\.)|\Z)', 
                             content, re.IGNORECASE | re.DOTALL)
        
        # Extract captions
        captions = re.findall(r'caption:\s*(.*?)(?=\n\s*(?:reason|start|end|clip|\d+\.)|\Z)', 
                              content, re.IGNORECASE | re.DOTALL)
        
        # Match up the extracted information
        for i in range(min(len(start_times), len(end_times))):
            clip = {
                "start": start_times[i],
                "end": end_times[i],
                "reason": reasons[i].strip() if i < len(reasons) else "Interesting moment",
                "caption": captions[i].strip() if i < len(captions) else "Check out this moment!"
            }
            clips.append(clip)
        
        return {"clips": clips}
    
    def _fallback_extraction(self, transcription_segments):
        """Simple fallback method if all API calls fail"""
        clips = []
        
        # Group segments into potential clips (simple approach)
        # This is a very basic fallback that just picks evenly spaced segments
        total_segments = len(transcription_segments)
        num_clips = min(5, total_segments // 3)  # Create up to 5 clips
        
        if num_clips == 0 and total_segments > 0:
            num_clips = 1
        
        for i in range(num_clips):
            idx = (i * total_segments) // num_clips
            segment = transcription_segments[idx]
            
            # Calculate clip start/end (aim for 45-60 second clips)
            clip_mid = (segment["start"] + segment["end"]) / 2
            clip_start = max(0, clip_mid - 25)
            clip_end = min(clip_mid + 25, segment["end"] + 30)
            
            clip = {
                "start": self._format_time(clip_start),
                "end": self._format_time(clip_end),
                "reason": "Potentially interesting segment",
                "caption": segment["text"][:100] + "..." if len(segment["text"]) > 100 else segment["text"]
            }
            clips.append(clip)
        
        return {"clips": clips}
    
    def _format_time(self, seconds):
        """Format seconds to mm:ss format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file"""
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_path} -y"
    subprocess.call(command, shell=True)
    return output_path


def transcribe_audio(audio_path, whisper_model_size="base"):
    """Transcribe audio using Whisper and return segments"""
    print("Loading Whisper model...")
    model = whisper.load_model(whisper_model_size)
    
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Extract segments from the result
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": segment.get("words", [])
        })
    
    # Process segments to create text lines for captions, similar to instaClips.py
    for segment in segments:
        clip_words = []
        for word in segment.get("words", []):
            clip_words.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
        
        # Create text lines for captions
        width = 1080  # Instagram width
        sample_text = "Sample Text for Calculation"
        font_size = 60  # Default font size
        
        # Create a temporary image to measure text size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate average char width
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        sample_width = bbox[2] - bbox[0]
        char_width = sample_width / len(sample_text)
        
        # Calculate usable width (80% of screen width)
        usable_width = int(width * 0.8)
        chars_per_line = int(usable_width / char_width)
        
        # Create text lines based on words
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
                "end": clip_words[-1]["end"] if clip_words else segment["end"]
            })
            
        # Add text_lines to segment
        segment["text_lines"] = text_lines
    
    return segments


def parse_timestamp(timestamp):
    """Convert 'mm:ss' timestamp to seconds"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def review_clips(clips, transcription_segments):
    """Allow user to review and edit clips before creating them"""
    approved_clips = []
    
    print("\n=== Clips to Review ===")
    for i, clip in enumerate(clips):
        print(f"\nClip {i+1}:")
        print(f"  Time: {clip['start']} to {clip['end']}")
        print(f"  Reason: {clip['reason']}")
        print(f"  Caption: {clip['caption']}")
        
        # Display transcript for this time period
        start_time = parse_timestamp(clip["start"])
        end_time = parse_timestamp(clip["end"])
        
        print("\n  Transcript:")
        relevant_text = []
        for segment in transcription_segments:
            if segment["end"] >= start_time and segment["start"] <= end_time:
                relevant_text.append(segment["text"])
        
        transcript = " ".join(relevant_text)
        wrapped_text = textwrap.fill(transcript, width=70)
        for line in wrapped_text.split("\n"):
            print(f"    {line}")
        
        # Ask for user action
        while True:
            action = input("\nActions: [a]pprove, [e]dit caption, [t]rim times, [s]kip): ").lower()
            
            if action == 'a':
                approved_clips.append(clip)
                print("Clip approved!")
                break
                
            elif action == 'e':
                new_caption = input(f"Enter new caption (current: {clip['caption']}): ")
                if new_caption:
                    clip["caption"] = new_caption
                    print("Caption updated.")
                approved_clips.append(clip)
                break
                
            elif action == 't':
                new_start = input(f"New start time (current: {clip['start']}, format mm:ss): ")
                if new_start:
                    clip["start"] = new_start
                
                new_end = input(f"New end time (current: {clip['end']}, format mm:ss): ")
                if new_end:
                    clip["end"] = new_end
                
                print(f"Clip timing updated: {clip['start']} to {clip['end']}")
                approved_clips.append(clip)
                break
                
            elif action == 's':
                print("Clip skipped.")
                break
                
            else:
                print("Invalid action. Please try again.")
    
    return approved_clips


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


def create_clip(video_path, clip, output_path, captions=True):
    """Create a video clip with optional captions using techniques from instaClips.py"""
    # Convert timestamps to seconds
    start_time = parse_timestamp(clip["start"])
    end_time = parse_timestamp(clip["end"])
    duration = end_time - start_time
    
    # Extract the clip from the original video with FFmpeg
    temp_video = f"{output_path}_temp.mp4"
    extract_cmd = f"ffmpeg -ss {start_time} -i {video_path} -t {duration} -c:v copy -c:a copy {temp_video} -y"
    print(f"Extracting clip: {extract_cmd}")
    subprocess.call(extract_cmd, shell=True)
    
    # Now open the extracted segment
    cap = cv2.VideoCapture(temp_video)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set output dimensions (9:16 aspect ratio for Instagram)
    target_width = 1080
    target_height = 1920
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(f"{output_path}_processed.mp4", fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        print("Failed to open video writer with H.264 codec. Trying MPEG-4...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{output_path}_processed.mp4", fourcc, fps, (target_width, target_height))
        
        if not out.isOpened():
            print("Could not open video writer with either codec. Check your OpenCV installation.")
            return None
    
    # Extract text lines from segments for captions
    text_lines = []
    if captions:
        # Find segments that overlap with this clip
        for segment in clip.get("segments", []):
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # Check if segment overlaps with clip
            if segment_end >= start_time and segment_start <= end_time:
                # Adjust text line timing relative to clip start
                for line in segment.get("text_lines", []):
                    text_lines.append({
                        "text": line["text"],
                        "start": max(0, line["start"] - start_time),
                        "end": min(duration, line["end"] - start_time)
                    })
        
        # If no segments provided, use the caption as a static text
        if not text_lines and clip.get("caption"):
            text_lines = [{
                "text": clip["caption"],
                "start": 0,
                "end": duration
            }]
    
    print(f"Processing {total_frames} frames at {fps} fps")
    
    # Process frame by frame
    frames_processed = 0
    while frames_processed < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Keep track of current time in the clip
        current_time = frames_processed / fps
        frames_processed += 1
        
        # Get original dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Create a 9:16 canvas
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Create blurred background (similar to instaClips.py)
        bg_scale = target_height / orig_height
        bg_width = int(orig_width * bg_scale)
        
        # Resize for background
        bg_resized = cv2.resize(frame, (bg_width, target_height))
        
        # Handle background sizing
        if bg_width > target_width:
            x_offset = (bg_width - target_width) // 2
            bg_resized = bg_resized[:, x_offset:x_offset+target_width]
        else:
            x_offset = (target_width - bg_width) // 2
            result[:, x_offset:x_offset+bg_width] = bg_resized
            bg_resized = result.copy()
        
        # Apply blur to background
        blurred_bg = cv2.GaussianBlur(bg_resized, (151, 151), 0)
        
        # Calculate video placement - using 70% of screen height as in instaClips.py
        video_height = int(target_height * 0.7)
        scale_factor = video_height / orig_height
        video_width = int(orig_width * scale_factor)
        
        # Ensure video width fits
        if video_width > target_width * 1.0:
            video_width = int(target_width * 1.0)
            scale_factor = video_width / orig_width
            video_height = int(orig_height * scale_factor)
        
        # Resize original frame to target size
        orig_resized = cv2.resize(frame, (video_width, video_height))
        
        # Center the video horizontally and vertically
        x_offset = (target_width - video_width) // 2
        y_offset = (target_height - video_height) // 2
        
        # Create final result with blurred background
        result = blurred_bg.copy()
        result[y_offset:y_offset+video_height, x_offset:x_offset+video_width] = orig_resized
        
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
            
            # Find the text line for the current time
            current_line = None
            for line in text_lines:
                if line["start"] <= current_time <= line["end"]:
                    current_line = line["text"]
                    break
            
            # If no exact match, use the last line that started before current time
            if not current_line and text_lines:
                candidates = [line for line in text_lines if line["start"] <= current_time]
                if candidates:
                    current_line = max(candidates, key=lambda x: x["start"])["text"]
            
            # If no text lines available, use clip caption
            if not current_line and clip.get("caption"):
                current_line = clip["caption"]
            
            if current_line:  # Only proceed if we have text to display
                # Calculate text size
                bbox = draw.textbbox((0, 0), current_line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position text below video
                text_x = (target_width - text_width) // 2
                text_y = y_offset + video_height + 60  # Increased spacing
                
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
    
    # Combine processed video with the audio from the extracted clip
    final_output = f"{output_path}"
    combine_cmd = f"ffmpeg -i {output_path}_processed.mp4 -i {temp_video} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {final_output} -y"
    print(f"Adding audio: {combine_cmd}")
    subprocess.call(combine_cmd, shell=True)
    
    # Clean up temporary files
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(f"{output_path}_processed.mp4"):
        os.remove(f"{output_path}_processed.mp4")
    
    # Verify output file was created successfully
    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        print(f"Successfully created clip at {final_output}")
        return final_output
    else:
        print(f"Failed to create clip at {final_output}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create video clips using AI to find interesting moments")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default="ai_clips", help="Directory to save output clips")
    parser.add_argument("--min-clips", type=int, default=3, help="Minimum number of clips to suggest")
    parser.add_argument("--max-clips", type=int, default=8, help="Maximum number of clips to suggest")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--api-key", help="API key for LLM service (optional)")
    parser.add_argument("--captions", action="store_true", help="Add captions to clips")
    parser.add_argument("--no-review", action="store_true", help="Skip clip review")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Step 1: Extract audio from video
    print("Extracting audio from video...")
    audio_path = extract_audio(args.video_path)
    
    # Step 2: Transcribe audio
    print("Transcribing audio...")
    transcription_segments = transcribe_audio(audio_path, args.whisper_model)
    
    # Save transcription to file
    transcription_path = os.path.join(args.output_dir, "transcription.json")
    with open(transcription_path, "w") as f:
        json.dump(transcription_segments, f, indent=2)
    
    print(f"Transcription saved to {transcription_path}")
    
    # Step 3: Find interesting clips using LLM
    print("Finding interesting moments using LLM...")
    clip_finder = LLMClipFinder(api_key=args.api_key)
    clip_suggestions = clip_finder.find_interesting_moments(
        transcription_segments, 
        min_clips=args.min_clips, 
        max_clips=args.max_clips
    )
    
    if not clip_suggestions or "clips" not in clip_suggestions or not clip_suggestions["clips"]:
        print("No interesting clips found. Exiting.")
        return
    
    clips = clip_suggestions["clips"]
    print(f"Found {len(clips)} potential clips")
    
    # Save clip suggestions to file
    suggestions_path = os.path.join(args.output_dir, "clip_suggestions.json")
    with open(suggestions_path, "w") as f:
        json.dump(clip_suggestions, f, indent=2)
    
    print(f"Clip suggestions saved to {suggestions_path}")
    
    # Enhance clips with segments for better captioning
    for clip in clips:
        clip_start = parse_timestamp(clip["start"])
        clip_end = parse_timestamp(clip["end"])
        
        # Find segments that overlap with this clip
        clip["segments"] = []
        for segment in transcription_segments:
            if segment["end"] >= clip_start and segment["start"] <= clip_end:
                clip["segments"].append(segment)
    
    # Step 4: Review clips if requested
    if not args.no_review:
        print("\nReviewing clips...")
        approved_clips = review_clips(clips, transcription_segments)
    else:
        approved_clips = clips
    
    if not approved_clips:
        print("No clips approved. Exiting.")
        return
    
    # Step 5: Create approved clips
    created_clips = []
    for i, clip in enumerate(approved_clips):
        print(f"\nCreating clip {i+1}/{len(approved_clips)}...")
        output_path = os.path.join(args.output_dir, f"clip_{i+1}.mp4")
        try:
            clip_path = create_clip(args.video_path, clip, output_path, captions=args.captions)
            if clip_path:
                created_clips.append(clip_path)
                print(f"Successfully created clip at {clip_path}")
            else:
                print(f"Failed to create clip {i+1}")
        except Exception as e:
            print(f"Error creating clip {i+1}: {str(e)}")
    
    # Step 6: Clean up and report results
    os.remove(audio_path)
    print(f"\nProcess complete! Created {len(created_clips)} clips in {args.output_dir}")
    
    # Save metadata about created clips
    clips_metadata = {
        "created_clips": [
            {
                "path": clip_path,
                "details": approved_clips[i]
            } for i, clip_path in enumerate(created_clips)
        ]
    }
    
    metadata_path = os.path.join(args.output_dir, "clips_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(clips_metadata, f, indent=2)


if __name__ == "__main__":
    main()