import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import whisper
import argparse
import textwrap
from typing import List, Dict, Any, Tuple
from collections import deque

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
    
    return segments


def review_transcription(transcription_segments):
    """Allow user to review and edit the transcription"""
    print("\n=== Transcription Review ===")
    print("Review each segment and correct any errors.")
    print("For each segment, you can:")
    print("  [enter] - Accept as is")
    print("  [edit text] - Type corrected text")
    print("  q - Finish review and save changes")
    print("  s - Skip to end without further review\n")
    
    for i, segment in enumerate(transcription_segments):
        print(f"\nSegment {i+1}/{len(transcription_segments)}")
        print(f"[{format_time(segment['start'])} - {format_time(segment['end'])}]")
        print(f"Current text: {segment['text']}")
        
        user_input = input("Correction (or enter to accept): ")
        
        if user_input.lower() == 'q':
            print("Finishing review...")
            break
        elif user_input.lower() == 's':
            print("Skipping remaining segments...")
            break
        elif user_input:
            # Update the segment text with correction
            transcription_segments[i]['text'] = user_input
            print(f"Updated: {user_input}")
            
            # Update words if they exist (simple approach - just reset timing)
            if 'words' in segment and segment['words']:
                avg_word_duration = (segment['end'] - segment['start']) / len(user_input.split())
                new_words = []
                words = user_input.split()
                current_time = segment['start']
                
                for word in words:
                    word_end = current_time + avg_word_duration
                    new_words.append({
                        "word": word,
                        "start": current_time,
                        "end": word_end
                    })
                    current_time = word_end
                
                transcription_segments[i]['words'] = new_words
    
    print("\nTranscription review complete!")
    return transcription_segments


def process_segments_for_captions(segments, video_width):
    """Process segments to create text lines for captions"""
    for segment in segments:
        clip_words = []
        for word in segment.get("words", []):
            clip_words.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
        font_path = "C:/Windows/Fonts/ARLRDBD.TTF"  # Adjust font path as needed
        font = font_path if os.path.exists(font_path) else "arial.ttf"
        
        # Calculate reasonable font size based on video width
        font_size = max(28, int(video_width * 0.03))
        
        # Create a temporary image to measure text size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate average char width
        sample_text = "Sample Text for Width Calculation"
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        sample_width = bbox[2] - bbox[0]
        char_width = sample_width / len(sample_text)
        
        # Calculate usable width (80% of screen width)
        usable_width = int(video_width * 0.8)
        chars_per_line = int(usable_width / char_width)
        
        # Create text lines based on words
        text_lines = []
        current_line = ""
        current_line_words = []
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
                        "words": current_line_words,
                        "start": line_start_time,
                        "end": word["start"]
                    })
                
                # Start new line with current word
                current_line = word_text
                current_line_words = [word]
                line_start_time = word["start"]
            else:
                # Add word to current line
                current_line = test_line
                current_line_words.append(word)
        
        # Add the last line if there is one
        if current_line:
            text_lines.append({
                "text": current_line,
                "words": current_line_words,
                "start": line_start_time,
                "end": clip_words[-1]["end"] if clip_words else segment["end"]
            })
            
        # Add text_lines to segment
        segment["text_lines"] = text_lines
    
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
    
    # Fix: Ensure x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        # Invalid rectangle, skip drawing
        return
    
    # Ensure radius isn't too large for the rectangle
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        # If radius is invalid, just draw a normal rectangle
        draw.rectangle((x1, y1, x2, y2), fill=fill)
        return
        
    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    # Draw four corners
    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)


def format_time(seconds):
    """Format seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def create_blurred_background(frame, x1, y1, x2, y2, blur_amount=15):
    """Create a blurred background from a region of the frame"""
    # Extract the region to blur
    region = frame[y1:y2, x1:x2]
    
    # Apply blur
    blurred = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
    
    # Return blurred region
    return blurred


def parse_color(color_str):
    """Parse color string in format 'r,g,b,a' or 'r,g,b'"""
    parts = color_str.split(',')
    if len(parts) == 3:
        # RGB format
        return tuple(int(p.strip()) for p in parts) + (255,)  # Default alpha to 255
    elif len(parts) == 4:
        # RGBA format
        return tuple(int(p.strip()) for p in parts)
    else:
        raise ValueError("Color must be in format 'r,g,b' or 'r,g,b,a'")


def caption_video(video_path, output_path, segments, bg_color=(255, 255, 255, 0), 
                 highlight_color=(255, 226, 165, 220), text_color=(0, 0, 0)):
    # Store the last few highlighted words for animation    
    last_highlighted_words = deque(maxlen=5)  # Stores previous highlighted positions
    animation_duration = 0.05  # Seconds for transition animation

    """Add captions to the entire video while preserving aspect ratio"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(f"{output_path}_temp.mp4", fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Failed to open video writer with H.264 codec. Trying MPEG-4...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{output_path}_temp.mp4", fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Could not open video writer with either codec. Check your OpenCV installation.")
            return None
    
    # Calculate font size based on video width
    font_size = max(28, int(width * 0.03))
    
    # Set caption padding
    bottom_padding = int(height * 0.05)
    
    # Process frame by frame
    frames_processed = 0
    last_text_end_time = 0  # Track when the last text segment ended
    silence_duration = 2.0  # Seconds of silence before hiding captions
    
    print(f"Processing {total_frames} frames at {fps} fps")
    
    while frames_processed < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Keep track of current time in the video
        current_time = frames_processed / fps
        frames_processed += 1
        
        if frames_processed % 100 == 0:
            print(f"Progress: {frames_processed}/{total_frames} frames ({frames_processed/total_frames*100:.1f}%)")
        
        # Find current active line(s)
        active_lines = []
        for segment in segments:
            # Check if segment has text_lines
            if "text_lines" in segment:
                for line in segment["text_lines"]:
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
            recent_lines = []
            for segment in segments:
                if "text_lines" in segment:
                    for line in segment["text_lines"]:
                        if line["end"] <= current_time:
                            recent_lines.append(line)
                            
            if recent_lines:
                most_recent = max(recent_lines, key=lambda x: x["end"])
                if current_time - most_recent["end"] < silence_duration:
                    active_lines = [most_recent]
        
        # Only proceed if we have lines to display and should show captions
        if active_lines and show_caption:
            # Convert CV2 image to PIL for text rendering
            pil_img = cv2_to_pil(frame)
            draw = ImageDraw.Draw(pil_img)
            font_path = "C:/Windows/Fonts/ARLRDBD.TTF"
            
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
            
            # Display each active line (up to 2 lines to avoid cluttering)
            for i, line in enumerate(active_lines[:2]):  # Limit to 2 lines max
                line_text = line["text"]
                line_words = line.get("words", [])
                
                # Calculate text size for positioning
                bbox = draw.textbbox((0, 0), line_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position text at bottom with spacing
                center_x = width // 2
                text_y = height - bottom_padding - (text_height * (2 - i))  # Arrange lines bottom-up
                
                # Background for text (semi-transparent)
                bg_padding = int(font_size * 0.5)  # Adjust padding based on font size
                corner_radius = int(font_size * 0.5)  # Adjust radius based on font size
                
                # Modified code to fix background rectangle:
                if line_words:
                    # Calculate width based on word positions
                    all_words_text = " ".join([w["text"] for w in line_words])
                    all_words_bbox = draw.textbbox((0, 0), all_words_text, font=font)
                    text_width = all_words_bbox[2] - all_words_bbox[0]
                else:
                    # Use the text line's bbox
                    text_bbox = draw.textbbox((0, 0), line_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]

                # Add extra safety padding
                safety_padding = int(font_size * 0.3)  # Extra padding to ensure all text is covered
                bg_width = text_width + (bg_padding * 2) + safety_padding
                bg_x_start = center_x - (bg_width / 2)
                bg_x_end = center_x + (bg_width / 2)

                caption_bg_bbox = (
                    bg_x_start,
                    text_y - bg_padding,
                    bg_x_end,
                    text_y + text_height + bg_padding
                )
                
                # Calculate frame coordinates for blur region (convert PIL coordinates to CV2)
                bg_y_start = int(text_y - bg_padding)
                bg_y_end = int(text_y + text_height + bg_padding)
                
                # Ensure coordinates are within frame boundaries
                bg_x_start_cv = max(0, int(bg_x_start))
                bg_x_end_cv = min(width, int(bg_x_end))
                bg_y_start_cv = max(0, bg_y_start)
                bg_y_end_cv = min(height, bg_y_end)
                
                # Convert back to CV2 temporarily to get the blurred region
                cv2_frame = pil_to_cv2(pil_img)
                
                # Apply blur to the background region
                blurred_region = cv2.GaussianBlur(
                    cv2_frame[bg_y_start_cv:bg_y_end_cv, bg_x_start_cv:bg_x_end_cv],
                    (21, 21), 0
                )
                
                # Place blurred region back into frame
                cv2_frame[bg_y_start_cv:bg_y_end_cv, bg_x_start_cv:bg_x_end_cv] = blurred_region
                
                # Convert back to PIL for drawing
                pil_img = cv2_to_pil(cv2_frame)
                draw = ImageDraw.Draw(pil_img)
                
                # Draw semi-transparent background
                draw_rounded_rectangle(draw, caption_bg_bbox, corner_radius, fill=bg_color)
                
                # First, determine dimensions of each word for highlighting
                if line_words:
                    word_positions = []
                    current_x = center_x - (text_width // 2)
                    
                    for word in line_words:
                        word_text = word["text"]
                        word_bbox = draw.textbbox((0, 0), word_text, font=font)
                        word_width = word_bbox[2] - word_bbox[0]
                        
                        # Add a little padding between words
                        if word_positions:  # Not the first word
                            space_bbox = draw.textbbox((0, 0), " ", font=font)
                            space_width = space_bbox[2] - space_bbox[0]
                            current_x += space_width
                        
                        word_positions.append({
                            "text": word_text,
                            "x": current_x,
                            "width": word_width,
                            "active": word["start"] <= current_time <= word["end"]
                        })
                        
                        current_x += word_width
                    
                    # Draw highlighted background for active words
                    current_active_words = [word_pos for word_pos in word_positions if word_pos["active"]]

                    # Update history of highlighted words with current active ones
                    if current_active_words:
                        # Add current frame's active words to history
                        for word in current_active_words:
                            word_info = {
                                "x": word["x"],
                                "width": word["width"],
                                "time": current_time
                            }
                            # Check if this word is already in the history
                            exists = False
                            for old_word in last_highlighted_words:
                                if abs(old_word["x"] - word_info["x"]) < 5:  # Same position
                                    exists = True
                                    old_word["time"] = current_time  # Update time
                                    break
                            if not exists:
                                last_highlighted_words.append(word_info)

                    # Remove old words from history
                    active_highlights = []
                    for word in list(last_highlighted_words):
                        time_diff = current_time - word["time"]
                        if time_diff > animation_duration:
                            last_highlighted_words.remove(word)
                        else:
                            active_highlights.append(word)

                    # Draw current word highlights and animate transitions
                    if current_active_words:
                        # Get the current active word position (rightmost word if multiple)
                        current_word = max(current_active_words, key=lambda w: w["x"])
                        
                        # If we have previous highlighted words, animate between them
                        previous_words = [w for w in active_highlights if w["time"] < current_time - 0.01]
                        
                        if previous_words:
                            # Find most recent previous highlighted word
                            prev_word = max(previous_words, key=lambda w: w["time"])
                            
                            # Calculate animation progress (0.0 to 1.0)
                            progress = min(1.0, (current_time - prev_word["time"]) / animation_duration)
                            
                            # Interpolate between previous position and current position
                            start_x = prev_word["x"]
                            start_width = prev_word["width"]
                            end_x = current_word["x"]
                            end_width = current_word["width"]
                            
                            # Lerp (linear interpolation) between positions
                            anim_x = start_x + (end_x - start_x) * progress
                            anim_width = start_width + (end_width - start_width) * progress
                            
                            # Draw animated highlight
                            highlight_padding = int(font_size * 0.2)
                            highlight_x1 = anim_x - highlight_padding // 2
                            highlight_x2 = anim_x + anim_width + highlight_padding // 2
                            
                            # Ensure x2 > x1
                            if highlight_x2 <= highlight_x1:
                                highlight_x2 = highlight_x1 + 2  # Minimum width
                            
                            highlight_bbox = (
                                highlight_x1,
                                text_y - highlight_padding // 2,
                                highlight_x2,
                                text_y + text_height + highlight_padding // 2
                            )
                            draw_rounded_rectangle(draw, highlight_bbox, int(font_size * 0.25), fill=highlight_color)
                        else:
                            # No previous word, just highlight current word normally
                            for word_pos in current_active_words:
                                highlight_padding = int(font_size * 0.2)
                                highlight_x1 = word_pos["x"] - highlight_padding // 2
                                highlight_x2 = word_pos["x"] + word_pos["width"] + highlight_padding // 2
                                
                                # Ensure x2 > x1
                                if highlight_x2 <= highlight_x1:
                                    highlight_x2 = highlight_x1 + 2  # Minimum width
                                    
                                highlight_bbox = (
                                    highlight_x1,
                                    text_y - highlight_padding // 2,
                                    highlight_x2,
                                    text_y + text_height + highlight_padding // 2
                                )
                                draw_rounded_rectangle(draw, highlight_bbox, int(font_size * 0.25), fill=highlight_color)                 
                    # Draw all words
                    for word_pos in word_positions:
                        draw.text(
                            (word_pos["x"], text_y),
                            word_pos["text"],
                            font=font,
                            fill=text_color
                        )
                else:
                    # If no word-level timing is available, just draw the whole text
                    text_x = center_x - (text_width // 2)
                    draw.text(
                        (text_x, text_y),
                        line_text,
                        font=font,
                        fill=text_color
                    )
            
            # Convert back to CV2 for video writing
            result = pil_to_cv2(pil_img)
            out.write(result)
        else:
            # No captions, just write the original frame
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processed {frames_processed} frames")
    
    # Add audio to the captioned video
    final_output = f"{output_path}"
    combine_cmd = f"ffmpeg -i {output_path}_temp.mp4 -i {video_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {final_output} -y"
    print(f"Adding audio: {combine_cmd}")
    subprocess.call(combine_cmd, shell=True)
    
    # Clean up temporary files
    if os.path.exists(f"{output_path}_temp.mp4"):
        os.remove(f"{output_path}_temp.mp4")
    
    # Verify output file was created successfully
    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        print(f"Successfully created captioned video at {final_output}")
        return final_output
    else:
        print(f"Failed to create captioned video at {final_output}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Caption an entire video with word-level subtitles")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-path", help="Path for the output video file", default="captioned_video.mp4")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--transcription-path", help="Path to existing transcription JSON (optional)")
    parser.add_argument("--skip-review", action="store_true", help="Skip transcription review")
    
    # Add color customization options
    parser.add_argument("--bg-color", default="255,255,255,0", 
                        help="Background color in format 'r,g,b,a' (e.g., '255,255,255,0')")
    parser.add_argument("--highlight-color", default="255,226,165,220", 
                        help="Highlight color in format 'r,g,b,a' (e.g., '255,226,165,220')")
    parser.add_argument("--text-color", default="0,0,0", 
                        help="Text color in format 'r,g,b' (e.g., '0,0,0')")
    
    args = parser.parse_args()
    
    # Parse color arguments
    try:
        bg_color = parse_color(args.bg_color)
        highlight_color = parse_color(args.highlight_color)
        text_color = parse_color(args.text_color)[:3]  # Text color doesn't need alpha
    except ValueError as e:
        print(f"Error parsing color: {e}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load existing transcription or create new one
    if args.transcription_path and os.path.exists(args.transcription_path):
        print(f"Loading existing transcription from {args.transcription_path}")
        with open(args.transcription_path, "r") as f:
            transcription_segments = json.load(f)
    else:
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio(args.video_path)
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription_segments = transcribe_audio(audio_path, args.whisper_model)
        
        # Save original transcription to file
        original_transcription_path = os.path.join(os.path.dirname(args.output_path), "original_transcription.json")
        with open(original_transcription_path, "w") as f:
            json.dump(transcription_segments, f, indent=2)
        
        print(f"Original transcription saved to {original_transcription_path}")
        
        # Clean up audio file
        os.remove(audio_path)
    
    # Review transcription if not skipped
    if not args.skip_review:
        print("\nReviewing transcription...")
        transcription_segments = review_transcription(transcription_segments)
        
        # Save reviewed transcription
        reviewed_transcription_path = os.path.join(os.path.dirname(args.output_path), "reviewed_transcription.json")
        with open(reviewed_transcription_path, "w") as f:
            json.dump(transcription_segments, f, indent=2)
        
        print(f"Reviewed transcription saved to {reviewed_transcription_path}")
    
    # Get video info to calculate dimensions for captions
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Process segments for captions
    print("Processing segments for captioning...")
    transcription_segments = process_segments_for_captions(transcription_segments, width)
    
    # Caption the video with custom colors
    print("Adding captions to video...")
    captioned_video = caption_video(
        args.video_path, 
        args.output_path, 
        transcription_segments,
        bg_color=bg_color,
        highlight_color=highlight_color,
        text_color=text_color
    )
    
    if captioned_video:
        print(f"\nProcess complete! Captioned video saved to {captioned_video}")
        # Print the color settings used
        print(f"Caption settings:")
        print(f"  Background color: {bg_color}")
        print(f"  Highlight color: {highlight_color}")
        print(f"  Text color: {text_color}")
    else:
        print("\nFailed to create captioned video.")


if __name__ == "__main__":
    main()