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
        
        while True:  # Continue until user decides to approve or skip
            # Display current clip info
            print(f"  Time: {clip['start']} to {clip['end']}")
            print(f"  Reason: {clip['reason']}")
            print(f"  Caption: {clip['caption']}")
            
            # Display transcript for this time period
            start_time = parse_timestamp(clip["start"])
            end_time = parse_timestamp(clip["end"])
            
            print("\n  Transcript:")
            relevant_segments = []
            for j, segment in enumerate(transcription_segments):
                if segment["end"] >= start_time and segment["start"] <= end_time:
                    relevant_segments.append((j, segment))
            
            # Display segments with index for reference
            for seg_idx, (trans_idx, segment) in enumerate(relevant_segments):
                print(f"    [{seg_idx}] {segment['text']}")
            
            # Ask for user action
            action = input("\nActions: [a]pprove, [e]dit transcript, [t]rim times, [s]kip, [n]ext clip: ").lower()
            
            if action == 'a':
                # Add a second buffer to both start and end time
                clip = add_time_buffer(clip, buffer_seconds=1)
                approved_clips.append(clip)
                print("Clip approved!")
                break
                
            elif action == 'e':
                # Edit transcript
                if relevant_segments:
                    seg_to_edit = input("Enter segment number to edit [0, 1, ...] or 'all' for all segments: ")
                    
                    if seg_to_edit.lower() == 'all':
                        # Edit all segments
                        for seg_idx, (trans_idx, segment) in enumerate(relevant_segments):
                            current_text = segment['text']
                            print(f"\nEditing segment [{seg_idx}]: {current_text}")
                            new_text = input("Enter corrected text (leave empty to keep unchanged): ")
                            
                            if new_text:
                                # Update in transcription_segments
                                transcription_segments[trans_idx]['text'] = new_text
                                print(f"Updated segment [{seg_idx}]")
                                
                                # Update text_lines as well if they exist
                                if 'text_lines' in transcription_segments[trans_idx]:
                                    # Create a simple one-line text_line
                                    transcription_segments[trans_idx]['text_lines'] = [{
                                        "text": new_text,
                                        "start": transcription_segments[trans_idx]["start"],
                                        "end": transcription_segments[trans_idx]["end"]
                                    }]
                    else:
                        try:
                            seg_idx = int(seg_to_edit)
                            if 0 <= seg_idx < len(relevant_segments):
                                trans_idx, segment = relevant_segments[seg_idx]
                                current_text = segment['text']
                                print(f"Current text: {current_text}")
                                new_text = input("Enter corrected text: ")
                                
                                if new_text:
                                    # Update in transcription_segments
                                    transcription_segments[trans_idx]['text'] = new_text
                                    print(f"Updated segment [{seg_idx}]")
                                    
                                    # Update text_lines as well if they exist
                                    if 'text_lines' in transcription_segments[trans_idx]:
                                        # Create a simple one-line text_line
                                        transcription_segments[trans_idx]['text_lines'] = [{
                                            "text": new_text,
                                            "start": transcription_segments[trans_idx]["start"],
                                            "end": transcription_segments[trans_idx]["end"]
                                        }]
                            else:
                                print("Invalid segment number.")
                        except ValueError:
                            print("Please enter a valid segment number or 'all'.")
                else:
                    print("No transcript segments available for this clip.")
                    
            elif action == 't':
                new_start = input(f"New start time (current: {clip['start']}, format mm:ss): ")
                if new_start:
                    clip["start"] = new_start
                
                new_end = input(f"New end time (current: {clip['end']}, format mm:ss): ")
                if new_end:
                    clip["end"] = new_end
                
                print(f"Clip timing updated: {clip['start']} to {clip['end']}")
                
            elif action == 's':
                print("Clip skipped.")
                break
                
            elif action == 'n':
                # Add a second buffer to both start and end time
                clip = add_time_buffer(clip, buffer_seconds=1)
                approved_clips.append(clip)
                print("Moving to next clip...")
                break
                
            else:
                print("Invalid action. Please try again.")
    
    return approved_clips, transcription_segments


def add_time_buffer(clip, buffer_seconds=1):
    """Add buffer time to clip start and end times"""
    # Parse current times
    start_time = parse_timestamp(clip["start"])
    end_time = parse_timestamp(clip["end"])
    
    # Add buffer (subtract from start, add to end)
    new_start_time = max(0, start_time - buffer_seconds)
    new_end_time = end_time + buffer_seconds
    
    # Format back to mm:ss
    clip["start"] = format_time(new_start_time)
    clip["end"] = format_time(new_end_time)
    
    return clip


def format_time(seconds):
    """Format seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


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
    """Create a video clip with all caps captions and word-by-word highlighting with proper line breaking"""
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
    
    # Process word timings and organize them into lines
    word_timings = []
    text_lines = []
    if captions:
        # First collect all words with their timings
        for segment in clip.get("segments", []):
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # Check if segment overlaps with clip
            if segment_end >= start_time and segment_start <= end_time:
                # Process words in this segment
                for word in segment.get("words", []):
                    # Only add words that will be in the clip timeframe
                    if word["end"] >= start_time and word["start"] <= end_time:
                        word_timings.append({
                            "text": word["word"].strip().upper(),  # Convert to uppercase
                            "start": max(0, word["start"] - start_time),
                            "end": min(duration, word["end"] - start_time)
                        })
        
        # If no words found, use entire caption as fallback
        if not word_timings and clip.get("caption"):
            word_timings = [{
                "text": clip["caption"].upper(),  # Convert to uppercase
                "start": 0,
                "end": duration
            }]
            
        # Now organize words into lines that fit on screen
        # Create a temporary PIL image to measure text size
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
            
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate usable width (80% of screen width)
        usable_width = int(target_width * 0.8)
        
        # Group words into lines
        current_line = []
        current_line_text = ""
        line_start_time = None
        
        for word in word_timings:
            word_text = word["text"]
            
            # Initialize line start time if needed
            if not current_line:
                line_start_time = word["start"]
            
            # Test if adding this word exceeds the line width
            test_line = current_line_text + (" " if current_line_text else "") + word_text
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width > usable_width and current_line:
                # Line would be too long, save current line and start a new one
                line_words = current_line.copy()
                line_end_time = line_words[-1]["end"]
                
                text_lines.append({
                    "words": line_words,
                    "text": current_line_text,
                    "start": line_start_time,
                    "end": line_end_time
                })
                
                # Start new line with current word
                current_line = [word]
                current_line_text = word_text
                line_start_time = word["start"]
            else:
                # Add word to current line
                current_line.append(word)
                current_line_text = test_line
        
        # Add the last line if it has content
        if current_line:
            text_lines.append({
                "words": current_line,
                "text": current_line_text,
                "start": line_start_time,
                "end": current_line[-1]["end"]
            })
    
    print(f"Processing {total_frames} frames at {fps} fps")
    
    # Process frame by frame
    frames_processed = 0
    last_text_end_time = 0  # Track when the last text segment ended
    silence_duration = 2.0  # Seconds of silence before hiding captions
    
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
        
        # Create blurred background
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
        
        # Calculate video placement - using 70% of screen height
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
            # Find current active line(s)
            active_lines = []
            for line in text_lines:
                # Line is active if it overlaps with the current time
                if line["start"] <= current_time <= line["end"]:
                    active_lines.append(line)
            
            # Update last text end time if there are active lines
            if active_lines:
                last_text_end_time = max([line["end"] for line in active_lines])
                show_caption = True
            elif current_time - last_text_end_time >= silence_duration:
                show_caption = False
            else:
                show_caption = True  # Still show caption during brief pauses
                
                # Find the most recent line that's finished
                recent_lines = [line for line in text_lines if line["end"] <= current_time]
                if recent_lines:
                    most_recent = max(recent_lines, key=lambda x: x["end"])
                    if current_time - most_recent["end"] < silence_duration:
                        active_lines = [most_recent]
            
            # Only proceed if we have lines to display and should show captions
            if active_lines and show_caption:
                # Convert CV2 image to PIL for text rendering
                pil_img = cv2_to_pil(result)
                draw = ImageDraw.Draw(pil_img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 60)
                except:
                    font = ImageFont.load_default()
                
                # Display each active line
                for i, line in enumerate(active_lines):
                    line_text = line["text"]
                    line_words = line["words"]
                    
                    # Calculate text size for positioning
                    bbox = draw.textbbox((0, 0), line_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position text below video with spacing between multiple lines
                    center_x = target_width // 2
                    text_y = y_offset + video_height + 60 + (i * (text_height + 30))  # Add vertical spacing between lines
                    
                    # Calculate the starting position for text
                    text_x = center_x - (text_width // 2)
                    
                    # First, determine dimensions of each word for highlighting
                    word_positions = []
                    current_x = text_x
                    
                    for word in line_words:
                        word_bbox = draw.textbbox((0, 0), word["text"], font=font)
                        word_width = word_bbox[2] - word_bbox[0]
                        
                        # Add a little padding between words
                        if word_positions:  # Not the first word
                            space_bbox = draw.textbbox((0, 0), " ", font=font)
                            space_width = space_bbox[2] - space_bbox[0]
                            current_x += space_width
                        
                        word_positions.append({
                            "text": word["text"],
                            "x": current_x,
                            "width": word_width,
                            "active": word["start"] <= current_time <= word["end"]
                        })
                        
                        current_x += word_width
                    
                    # Draw background for entire text area (semi-transparent white)
                    bg_padding = 40
                    corner_radius = 30

                    # Calculate centered background position
                    bg_width = text_width + (bg_padding * 2)
                    bg_x_start = center_x - (bg_width / 2)
                    bg_x_end = center_x + (bg_width / 2)

                    caption_bg_bbox = (
                        bg_x_start,
                        text_y - bg_padding,
                        bg_x_end,
                        text_y + text_height + bg_padding
                    )
                    draw_rounded_rectangle(draw, caption_bg_bbox, corner_radius, fill=(255, 255, 255, 230))
                    
                    # Draw highlighted background for active words
                    for word_pos in word_positions:
                        if word_pos["active"]:
                            # Draw yellow highlight behind active word
                            highlight_padding = 20
                            highlight_bbox = (
                                word_pos["x"] - highlight_padding // 2,
                                text_y - highlight_padding // 2,
                                word_pos["x"] + word_pos["width"] + highlight_padding // 2,
                                text_y + text_height + highlight_padding // 2.2
                            )
                            draw_rounded_rectangle(draw, highlight_bbox, 15, fill=(255, 226, 165, 220))  # Yellow highlight
                    
                    # Draw all words in black
                    for word_pos in word_positions:
                        draw.text(
                            (word_pos["x"], text_y),
                            word_pos["text"],
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
        approved_clips, updated_transcription = review_clips(clips, transcription_segments)
        
        # Save updated transcription
        with open(transcription_path, "w") as f:
            json.dump(updated_transcription, f, indent=2)
        print(f"Updated transcription saved to {transcription_path}")
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
            # Make sure to update segments in each clip with the latest transcription
            if not args.no_review:
                clip_start = parse_timestamp(clip["start"])
                clip_end = parse_timestamp(clip["end"])
                clip["segments"] = []
                for segment in updated_transcription:
                    if segment["end"] >= clip_start and segment["start"] <= clip_end:
                        clip["segments"].append(segment)
                
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