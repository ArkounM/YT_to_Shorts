import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont # Keep PIL for specific drawing if needed beyond caption_utils
import json
import argparse
import time # For simple delays or timing, if any
import textwrap # Keep for simple text wrapping if not fully covered by caption_utils logic
from typing import List, Dict, Any, Optional, Tuple
import copy # For deep copying segments

# Core utilities
from core.video_utils import extract_audio, get_video_dimensions
from core.transcription_utils import transcribe_audio, review_transcription as review_transcription_segments, set_transcription_device, format_time as format_time_hhmmss
from core.llm_utils import get_llm_provider, LLMProvider
from core.caption_utils import (
    process_segments_for_caption_lines,
    render_captions_on_frame,
    parse_color as parse_color_util,
    get_font_path as get_font_path_util,
    cv2_to_pil, # Exposing these if direct PIL manipulation is still chosen for some parts
    pil_to_cv2,
    draw_rounded_rectangle as draw_rounded_rectangle_util
)


class ClipFinder:
    """Handles finding interesting clips using a specified LLM provider."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def _format_time_for_llm(self, seconds: float) -> str:
        """Formats seconds to MM:SS for LLM prompts (can be part of LLMProvider too)."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def find_interesting_moments(self,
                                 transcription_segments: List[Dict[str, Any]],
                                 min_clips: int = 3,
                                 max_clips: int = 10,
                                 video_duration_seconds: Optional[float] = None
                                 ) -> Optional[Dict[str, Any]]:
        """Uses LLM to identify interesting moments."""

        transcript_text_for_llm = ""
        for segment in transcription_segments:
            # Use the LLM provider's time formatting or a local one
            start_time_str = self._format_time_for_llm(segment["start"])
            # end_time_str = self._format_time_for_llm(segment["end"])
            # transcript_text_for_llm += f"[{start_time_str}-{end_time_str}] {segment['text']}\n"
            transcript_text_for_llm += f"[{start_time_str}] {segment['text']}\n"


        duration_info = f"The video is approximately {video_duration_seconds / 60:.1f} minutes long." if video_duration_seconds else "The video duration is unknown."

        prompt = f"""
You are an expert video editor tasked with identifying compelling short clips from a video transcript.
{duration_info}
The transcript below includes timestamps in [MM:SS] format.

TRANSCRIPT:
{transcript_text_for_llm}

TASK:
Identify {min_clips}-{max_clips} moments that would make engaging short video clips, each ideally between 30 to 90 seconds.
Focus on:
1.  Interesting statements, stories, or punchlines.
2.  Emotional peaks (positive or negative).
3.  Surprising revelations, insights, or questions.
4.  Quotable or memorable phrases/segments.
5.  Self-contained ideas or points that are understandable in isolation.

OUTPUT FORMAT:
Return your response as a JSON object with a single key "clips".
The value of "clips" should be a list of objects, each with:
-   "start_time": string (MM:SS format, e.g., "01:23") - when the interesting moment begins.
-   "end_time": string (MM:SS format, e.g., "01:58") - when the interesting moment ends.
-   "reason": string (briefly why this moment is compelling, 1-2 sentences).
-   "suggested_caption": string (a short, catchy caption for this clip, ideally under 150 characters).

Example:
{{
  "clips": [
    {{
      "start_time": "02:15",
      "end_time": "02:45",
      "reason": "Speaker makes a very surprising and controversial statement here.",
      "suggested_caption": "Wait, did they REALLY just say that?! 🤯 #shocking"
    }}
  ]
}}
Ensure all timestamps are in MM:SS format. The duration between start_time and end_time should be reasonable for a short clip.
Consider the flow of conversation; a clip should ideally have a clear beginning and end.
"""
        print("Sending clip suggestion prompt to LLM...")
        clip_data = self.llm_provider.generate_json(prompt) # Assumes generate_json is robust

        if clip_data and "clips" in clip_data and isinstance(clip_data["clips"], list):
            # Basic validation of clip format
            valid_clips = []
            for c in clip_data["clips"]:
                if (isinstance(c, dict) and
                    "start_time" in c and "end_time" in c and
                    "reason" in c and "suggested_caption" in c and
                    isinstance(c["start_time"], str) and isinstance(c["end_time"], str) # Further type checks
                   ):
                    # Attempt to parse timestamps to ensure validity early on
                    try:
                        parse_timestamp(c["start_time"])
                        parse_timestamp(c["end_time"])
                        valid_clips.append(c)
                    except ValueError:
                        print(f"Warning: Skipping clip due to invalid time format: {c}")
                else:
                    print(f"Warning: Skipping malformed clip data from LLM: {c}")
            
            if not valid_clips and clip_data["clips"]:
                print("LLM returned clip data, but none were in the expected format after validation.")
                return self._fallback_clip_extraction(transcription_segments, num_clips_target=min_clips, video_duration_seconds=video_duration_seconds)
            elif not valid_clips :
                 print("LLM did not return valid clip data. Attempting fallback.")
                 return self._fallback_clip_extraction(transcription_segments, num_clips_target=min_clips, video_duration_seconds=video_duration_seconds)

            clip_data["clips"] = valid_clips
            return clip_data
        else:
            print("LLM response for clips was not in the expected JSON format or missing 'clips' list. Attempting fallback.")
            return self._fallback_clip_extraction(transcription_segments, num_clips_target=min_clips, video_duration_seconds=video_duration_seconds)

    def _fallback_clip_extraction(self,
                                  transcription_segments: List[Dict[str, Any]],
                                  num_clips_target: int = 3,
                                  clip_duration_target_seconds: int = 60,
                                  video_duration_seconds: Optional[float] = None
                                  ) -> Dict[str, List[Dict[str, str]]]:
        """Generates some basic clip suggestions as a fallback."""
        print(f"Using fallback clip generation method (first {num_clips_target} segments or time-based).")
        clips = []
        if not transcription_segments:
            return {"clips": []}

        # If video duration is known, try to spread clips
        if video_duration_seconds and video_duration_seconds > clip_duration_target_seconds:
            num_possible_clips = int(video_duration_seconds // clip_duration_target_seconds)
            actual_num_clips = min(num_clips_target, num_possible_clips, len(transcription_segments))
            
            if actual_num_clips == 0 and len(transcription_segments) > 0: # Ensure at least one clip if segments exist
                actual_num_clips = 1

            for i in range(actual_num_clips):
                # Try to pick segments spread out over the video
                start_index = (len(transcription_segments) // actual_num_clips) * i
                segment = transcription_segments[start_index]
                
                clip_start_s = segment["start"]
                # Ensure end time doesn't exceed video duration or segment end significantly
                clip_end_s = min(clip_start_s + clip_duration_target_seconds, segment["end"] + 15, video_duration_seconds)
                
                # Make sure end is after start
                if clip_end_s <= clip_start_s : clip_end_s = clip_start_s + 15 # min 15s clip

                clips.append({
                    "start_time": self._format_time_for_llm(clip_start_s),
                    "end_time": self._format_time_for_llm(clip_end_s),
                    "reason": "Fallback: Potentially interesting segment based on spread.",
                    "suggested_caption": segment["text"][:100] + "..." if len(segment["text"]) > 100 else segment["text"]
                })
        else: # Fallback if duration is unknown or too short
            for i, segment in enumerate(transcription_segments):
                if i >= num_clips_target:
                    break
                clip_start_s = segment["start"]
                # Default to segment end, or a fixed duration if segment is too long
                clip_end_s = min(segment["end"], clip_start_s + clip_duration_target_seconds)
                if clip_end_s <= clip_start_s : clip_end_s = clip_start_s + 15

                clips.append({
                    "start_time": self._format_time_for_llm(clip_start_s),
                    "end_time": self._format_time_for_llm(clip_end_s),
                    "reason": "Fallback: Early segment in transcription.",
                    "suggested_caption": segment["text"][:100] + "..." if len(segment["text"]) > 100 else segment["text"]
                })
        
        return {"clips": clips}


def parse_timestamp(timestamp_str: str) -> float:
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
                                    # Create a better word-level representation
                                    words = new_text.split()
                                    seg_duration = transcription_segments[trans_idx]["end"] - transcription_segments[trans_idx]["start"]
                                    word_duration = seg_duration / len(words) if words else 0
                                    
                                    new_words = []
                                    for i, word in enumerate(words):
                                        word_start = transcription_segments[trans_idx]["start"] + (i * word_duration)
                                        word_end = word_start + word_duration
                                        new_words.append({
                                            "word": word,
                                            "start": word_start,
                                            "end": word_end
                                        })
                                    
                                    # Update the words in the segment
                                    transcription_segments[trans_idx]['words'] = new_words
                                    
                                    # Update text_lines with evenly distributed words
                                    lines = textwrap.wrap(new_text, width=40)  # Basic line breaking
                                    line_count = len(lines)
                                    line_duration = seg_duration / line_count if line_count else 0
                                    
                                    new_text_lines = []
                                    for i, line in enumerate(lines):
                                        line_start = transcription_segments[trans_idx]["start"] + (i * line_duration)
                                        line_end = line_start + line_duration
                                        new_text_lines.append({
                                            "text": line,
                                            "start": line_start,
                                            "end": line_end
                                        })
    
                                        transcription_segments[trans_idx]['text_lines'] = new_text_lines
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


def create_clip(video_path, clip, output_path, captions=True, bg_color=(255, 255, 255, 230), 
                highlight_color=(255, 226, 165, 220), text_color=(0, 0, 0)):
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
                            "text": word["word"].strip(), #.upper(),  # Convert to uppercase
                            "start": max(0, word["start"] - start_time),
                            "end": min(duration, word["end"] - start_time)
                        })
        
        # If no words found, use entire caption as fallback
        if not word_timings and clip.get("caption"):
            word_timings = [{
                "text": clip["caption"], #.upper(),  # Convert to uppercase
                "start": 0,
                "end": duration
            }]
            
        font_path = "C:/Windows/Fonts/ARLRDBD.TTF"  # Adjust font path as needed
        font = font_path if os.path.exists(font_path) else "arial.ttf"
        # Now organize words into lines that fit on screen
        # Create a temporary PIL image to measure text size
        try:
            font = ImageFont.truetype(font_path, 60)
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

        if frames_processed % 100 == 0:
            print(f"Progress: {frames_processed}/{total_frames} frames ({frames_processed/total_frames*100:.1f}%)")
        
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

                font_path = "C:/Windows/Fonts/ARLRDBD.TTF"  # Adjust font path as needed
                font = font_path if os.path.exists(font_path) else "arial.ttf"
                try:
                    font = ImageFont.truetype(font_path, 60)
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
                    
                    # Draw background for entire text area (using custom bg_color)
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
                    draw_rounded_rectangle(draw, caption_bg_bbox, corner_radius, fill=bg_color)
                    
                    # Draw highlighted background for active words
                    for word_pos in word_positions:
                        if word_pos["active"]:
                            # Draw highlight behind active word using custom highlight_color
                            highlight_padding = 20
                            highlight_bbox = (
                                word_pos["x"] - highlight_padding // 2,
                                text_y - highlight_padding // 2,
                                word_pos["x"] + word_pos["width"] + highlight_padding // 2,
                                text_y + text_height + highlight_padding // 2.2
                            )
                            draw_rounded_rectangle(draw, highlight_bbox, 15, fill=highlight_color)
                    
                    # Draw all words using custom text_color
                    for word_pos in word_positions:
                        draw.text(
                            (word_pos["x"], text_y),
                            word_pos["text"],
                            font=font,
                            fill=text_color
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
                        help="Whisper model size for transcription.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Device for Whisper transcription (cpu or gpu).")
    parser.add_argument("--transcription-path", help="Path to existing transcription JSON file (optional).")

    # LLM arguments
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "openai", "openrouter", "deepseek"],
                        help="LLM provider to use for clip suggestions.")
    parser.add_argument("--llm-model", default=None, help="Specific LLM model name to use (optional).")
    parser.add_argument("--api-key", help="API key for the selected LLM service.")

    # Clip behavior arguments
    parser.add_argument("--captions", action="store_true", help="Add captions to the generated clips.")
    parser.add_argument("--no-review", action="store_true", help="Skip interactive review of suggested clips.")

    # Caption styling arguments
    parser.add_argument("--font-path", default=None, help="Path to .ttf or .otf font file for captions.")
    parser.add_argument("--bg-color", default="255,255,255,230",
                        help="Caption background color 'R,G,B,A' (default: 255,255,255,230).")
    parser.add_argument("--highlight-color", default="255,226,165,220",
                        help="Caption highlight color 'R,G,B,A' (default: 255,226,165,220).")
    parser.add_argument("--text-color", default="0,0,0,255",
                        help="Caption text color 'R,G,B,A' (default: 0,0,0,255).")
    
    args = parser.parse_args()
    
    # --- 0. Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    set_transcription_device(args.device)
    
    # Validate color arguments early
    try:
        parse_color_util(args.bg_color)
        parse_color_util(args.highlight_color)
        parse_color_util(args.text_color)
    except ValueError as e:
        print(f"Error: Invalid color format provided: {e}")
        return

    # --- 1. Transcription ---
    transcription_segments = None
    if args.transcription_path and os.path.exists(args.transcription_path):
        print(f"Loading existing transcription from: {args.transcription_path}")
        with open(args.transcription_path, "r", encoding="utf-8") as f:
            transcription_segments = json.load(f)
    else:
        print(f"Starting transcription for {args.video_path}...")
        temp_audio_path = os.path.join(args.output_dir, "temp_clips_audio.wav")
        audio_path = extract_audio(args.video_path, output_path=temp_audio_path)
        if not audio_path:
            print("Audio extraction failed. Cannot generate clips.")
            return

        print(f"Transcribing audio with Whisper model '{args.whisper_model}' on device '{args.device}'.")
        # Word timestamps are crucial for good captioning on clips
        transcription_segments = transcribe_audio(audio_path, args.whisper_model, word_timestamps=True)

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
            except OSError as e:
                print(f"Warning: Could not delete temporary audio file: {e}")

        # Save the new transcription
        current_transcription_path = os.path.join(args.output_dir, "transcription_for_clips.json")
        with open(current_transcription_path, "w", encoding="utf-8") as f:
            json.dump(transcription_segments, f, indent=2, ensure_ascii=False)
        print(f"Transcription saved to: {current_transcription_path}")

    if not transcription_segments:
        print("No transcription data available. Cannot find or create clips.")
        return
    
    # Get video duration for LLM context if not already available
    # This could also come from transcription metadata if Whisper provides it.
    # For now, explicitly get it.
    video_duration = get_video_dimensions(args.video_path)[0] # Placeholder, ideally get actual duration
    # A better way to get duration:
    # from core.video_utils import get_video_duration as get_duration_seconds
    # video_duration_val = get_duration_seconds(args.video_path)


    # --- 2. Find Interesting Clips using LLM ---
    print(f"Initializing LLM provider: {args.llm_provider} (Model: {args.llm_model or 'default'})")
    try:
        llm_service = get_llm_provider(args.llm_provider, api_key=args.api_key, model=args.llm_model)
    except (ValueError, ImportError, ConnectionError) as e:
        print(f"Error initializing LLM provider: {e}")
        return

    clip_finder_instance = ClipFinder(llm_service)
    print("Finding interesting moments with LLM...")
    # Pass video_duration to LLM if available and useful for the prompt
    clip_suggestions_data = clip_finder_instance.find_interesting_moments(
        transcription_segments, 
        min_clips=args.min_clips, 
        max_clips=args.max_clips,
        # video_duration_seconds=video_duration_val # Pass if obtained
    )

    if not clip_suggestions_data or "clips" not in clip_suggestions_data or not clip_suggestions_data["clips"]:
        print("LLM (or fallback) did not return any clip suggestions. Exiting.")
        return
    
    suggested_clips_from_llm = clip_suggestions_data["clips"]
    print(f"LLM suggested {len(suggested_clips_from_llm)} potential clips.")

    raw_suggestions_path = os.path.join(args.output_dir, "clip_suggestions_raw.json")
    with open(raw_suggestions_path, "w", encoding="utf-8") as f:
        json.dump(clip_suggestions_data, f, indent=2, ensure_ascii=False)
    print(f"Raw clip suggestions saved to: {raw_suggestions_path}")

    # --- 3. Prepare Segments for Each Suggested Clip (for captioning and review) ---
    # This step ensures each suggested clip has the relevant transcription segments associated with it.
    # The 'segments' key within each clip dict will hold these.
    for clip_info in suggested_clips_from_llm:
        clip_start_s = parse_timestamp(clip_info["start_time"])
        clip_end_s = parse_timestamp(clip_info["end_time"])
        clip_info["segments_for_clip"] = []
        for original_segment in transcription_segments:
            # Check for overlap between original segment and clip time range
            if max(original_segment["start"], clip_start_s) < min(original_segment["end"], clip_end_s):
                # Deep copy to avoid modifying the main transcription_segments list if segments are trimmed
                segment_copy = copy.deepcopy(original_segment)
                # Optional: Adjust segment_copy start/end times to be relative to clip_start_s if needed by create_clip
                # Or, ensure create_clip handles absolute timestamps from original_segment and extracts accordingly.
                clip_info["segments_for_clip"].append(segment_copy)

    # --- 4. Review Clips (Optional) ---
    final_clips_to_create = suggested_clips_from_llm
    # The review_clips function needs to be adapted or a new one created for this script's specific needs.
    # It should allow editing start/end times, reason, caption, and potentially re-evaluating segments.
    if not args.no_review:
        print("\nStarting interactive clip review process...")
        # `review_clips` needs to be defined or imported and adapted.
        # It should take `suggested_clips_from_llm` and `transcription_segments` (for context).
        # It should return the list of approved/edited clips.
        # For now, assuming a placeholder or that it will be implemented:
        # final_clips_to_create, updated_transcription_segments = review_clips_interactive(suggested_clips_from_llm, transcription_segments)

        # Placeholder for review function:
        print("Clip review step is placeholder. Proceeding with all suggested clips.")
        final_clips_to_create = suggested_clips_from_llm # Assuming all are approved for now
        # If review_clips modified transcription_segments, ensure it's saved:
        # if updated_transcription_segments and args.transcription_path:
        #     with open(args.transcription_path, "w", encoding="utf-8") as f: # or a new "reviewed_transcription.json"
        #        json.dump(updated_transcription_segments, f, indent=2, ensure_ascii=False)
        #     print(f"Updated transcription saved after review.")

    if not final_clips_to_create:
        print("No clips approved or available after review. Exiting.")
        return

    # --- 5. Create Approved Clips ---
    created_clip_paths = []
    for i, clip_details in enumerate(final_clips_to_create):
        clip_num = i + 1
        print(f"\nProcessing clip {clip_num}/{len(final_clips_to_create)}: {clip_details.get('suggested_caption', 'Untitled Clip')}")

        clip_output_filename = f"clip_{clip_num}.mp4"
        clip_final_output_path = os.path.join(args.output_dir, clip_output_filename)

        try:
            # `create_clip` is the function that does the FFMPEG extraction and caption rendering
            # It needs to be adapted to use the new utility functions.
            generated_clip_filepath = create_clip_video(
                original_video_path=args.video_path,
                clip_info=clip_details, # Contains start_time, end_time, segments_for_clip
                output_path=clip_final_output_path,
                add_captions=args.captions,
                font_path_override=args.font_path,
                bg_color_str=args.bg_color,
                highlight_color_str=args.highlight_color,
                text_color_str=args.text_color
                # target_aspect_ratio can be added: e.g., (9,16) for vertical
            )
            if generated_clip_filepath:
                created_clip_paths.append(generated_clip_filepath)
                print(f"Successfully created clip: {generated_clip_filepath}")
            else:
                print(f"Failed to create video for clip {clip_num}.")
        except Exception as e:
            print(f"Error during creation of clip {clip_num}: {e}")
            import traceback
            traceback.print_exc()

    # --- 6. Report Results ---
    print(f"\n--- Process Summary ---")
    if created_clip_paths:
        print(f"Successfully created {len(created_clip_paths)} clips in directory: {args.output_dir}")
        for path in created_clip_paths:
            print(f"  - {path}")
    else:
        print("No clips were successfully created.")

    # Save metadata about the final created clips and their original suggestions
    final_metadata = {
        "source_video": args.video_path,
        "output_directory": args.output_dir,
        "llm_provider_used": args.llm_provider,
        "clips_generated": []
    }
    for i, clip_path in enumerate(created_clip_paths):
        # Find the corresponding clip_details (assuming order is maintained)
        original_clip_suggestion = final_clips_to_create[i] if i < len(final_clips_to_create) else {}
        final_metadata["clips_generated"].append({
            "output_file": clip_path,
            "original_suggestion": original_clip_suggestion # Includes reason, LLM caption, times
        })
    
    metadata_filepath = os.path.join(args.output_dir, "generated_clips_metadata.json")
    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata for generated clips saved to: {metadata_filepath}")
    print("-----------------------")

if __name__ == "__main__":
    main()