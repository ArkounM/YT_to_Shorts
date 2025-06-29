import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

# --- Font Handling ---
DEFAULT_FONT_PATH = "arial.ttf" # Default fallback
# Attempt to find a more suitable default font if available
PREFERRED_FONT_PATHS = [
    "C:/Windows/Fonts/ARLRDBD.TTF", # Arial Rounded MT Bold (used in original script)
    "C:/Windows/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf"
]

def get_font_path(font_path_override: Optional[str] = None, font_size: int = 28) -> ImageFont.FreeTypeFont:
    """Loads a font, trying specified path, then preferred paths, then default."""
    paths_to_try = []
    if font_path_override and os.path.exists(font_path_override):
        paths_to_try.append(font_path_override)

    paths_to_try.extend(p for p in PREFERRED_FONT_PATHS if os.path.exists(p))

    for path in paths_to_try:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    # Fallback to default PIL font if all else fails
    print(f"Warning: Specified or preferred fonts not found. Using PIL default font for size {font_size}.")
    try:
        return ImageFont.truetype(DEFAULT_FONT_PATH, font_size) # Try system Arial as a last resort truetype
    except:
        return ImageFont.load_default()


# --- Image Conversion Utilities ---
def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Drawing Utilities ---
def draw_rounded_rectangle(draw: ImageDraw.ImageDraw, bbox: Tuple[float, float, float, float], radius: int, fill: Tuple[int, int, int, int]):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1: # Invalid rectangle
        return

    # Ensure radius isn't too large
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius < 0: radius = 0 # Ensure radius is not negative

    if radius == 0: # If radius is zero, draw a normal rectangle
        draw.rectangle((x1,y1,x2,y2), fill=fill)
        return

    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)

    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)

def parse_color(color_str: str, default_alpha: int = 255) -> Tuple[int, int, int, int]:
    """Parse color string in format 'r,g,b,a' or 'r,g,b' into (R,G,B,A) tuple."""
    parts = [p.strip() for p in color_str.split(',')]
    if len(parts) == 3: # RGB
        return (int(parts[0]), int(parts[1]), int(parts[2]), default_alpha)
    elif len(parts) == 4: # RGBA
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    else:
        raise ValueError("Color must be in format 'r,g,b' or 'r,g,b,a'")


# --- Caption Processing Logic ---
def process_segments_for_caption_lines(
    segments: List[Dict[str, Any]],
    video_width: int,
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    chars_per_line_ratio: float = 0.8, # Ratio of video_width for usable text width
    max_lines_on_screen: int = 2
) -> List[Dict[str, Any]]:
    """
    Process transcription segments to create text lines for captions,
    considering video width and font metrics for line breaking.
    Each segment can be broken into multiple display lines.
    """
    if not font_size:
        font_size = max(28, int(video_width * 0.03)) # Default font size calculation

    font = get_font_path(font_path, font_size)

    # Create a temporary image to measure text size
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Calculate average char width more robustly
    sample_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        sample_width = bbox[2] - bbox[0]
        char_width = sample_width / len(sample_text) if len(sample_text) > 0 else font_size / 2 # fallback
    except Exception as e:
        print(f"Warning: Could not accurately measure text width using provided font. Using fallback. Error: {e}")
        char_width = font_size / 2 # Rough estimate if textbbox fails

    usable_width = int(video_width * chars_per_line_ratio)
    chars_per_line_estimate = int(usable_width / char_width) if char_width > 0 else 30 # Fallback chars per line

    processed_segments = []
    for segment in segments:
        segment_text_lines = []

        # Use pre-existing words if available and they have text
        segment_words = [word for word in segment.get("words", []) if word.get("word","").strip()]

        if not segment_words: # Fallback if no word timestamps, use segment text
            # Simple split by space, does not preserve original word timings
            words_from_text = segment["text"].split()
            if not words_from_text: # Skip if segment text is also empty
                processed_segments.append({**segment, "text_lines": []})
                continue

            # Distribute segment time over these words
            num_words_in_segment = len(words_from_text)
            segment_duration = segment["end"] - segment["start"]
            avg_word_duration = segment_duration / num_words_in_segment if num_words_in_segment > 0 else 0.1

            current_word_start_time = segment["start"]
            segment_words = []
            for word_text in words_from_text:
                word_end_time = current_word_start_time + avg_word_duration
                segment_words.append({
                    "text": word_text, # Note: original script used word["word"], so changed to "text" for consistency
                    "start": current_word_start_time,
                    "end": word_end_time
                })
                current_word_start_time = word_end_time

        current_line_words_data = []
        current_line_text_content = ""
        line_start_time = None

        for word_data in segment_words:
            word_text = word_data["text"].strip() # Ensure "text" key is used, matching Whisper output
            if not word_text:
                continue

            if line_start_time is None: # Start of a new line
                line_start_time = word_data["start"]

            test_line = current_line_text_content + " " + word_text if current_line_text_content else word_text

            # Measure actual width of test_line
            try:
                line_bbox = draw.textbbox((0,0), test_line, font=font)
                current_line_width = line_bbox[2] - line_bbox[0]
            except: # Fallback if font rendering fails for specific text
                current_line_width = len(test_line) * char_width


            if current_line_width > usable_width and current_line_words_data:
                # Current line is full, finalize it
                segment_text_lines.append({
                    "text": current_line_text_content,
                    "words": current_line_words_data,
                    "start": line_start_time,
                    "end": current_line_words_data[-1]["end"] # End time of the last word in the line
                })

                # Start a new line with the current word
                current_line_words_data = [word_data]
                current_line_text_content = word_text
                line_start_time = word_data["start"]
            else:
                # Add word to current line
                current_line_words_data.append(word_data)
                current_line_text_content = test_line

        # Add the last accumulated line
        if current_line_words_data:
            segment_text_lines.append({
                "text": current_line_text_content,
                "words": current_line_words_data,
                "start": line_start_time,
                "end": current_line_words_data[-1]["end"]
            })

        # Store these processed lines back into a copy of the segment
        processed_segments.append({**segment, "text_lines": segment_text_lines})

    return processed_segments


# --- Main Caption Rendering Function (Simplified version of generateCaptions.caption_video) ---
# This will be a complex function. For brevity in this step, I'll sketch its structure.
# The full implementation will be based on generateCaptions.py:caption_video
# and generateClips.py:create_clip's captioning parts.

def render_captions_on_frame(
    frame: np.ndarray,
    current_time: float,
    processed_segments: List[Dict[str, Any]],
    video_width: int,
    video_height: int,
    font_path_override: Optional[str] = None,
    font_size_ratio: float = 0.03, # of video_width
    bottom_padding_ratio: float = 0.05, # of video_height
    bg_color_str: str = "255,255,255,0", # transparent white
    highlight_color_str: str = "255,226,165,220", # semi-transparent yellow
    text_color_str: str = "0,0,0,255", # opaque black
    max_lines_on_screen: int = 2,
    line_spacing: int = 10, # pixels
    bg_padding: int = 15, # pixels around text for background
    corner_radius: int = 10, # for rounded background
    animation_duration: float = 0.05, # for word highlight transition
    last_highlighted_words_deque: Optional[deque] = None # Pass deque for stateful animation
    ) -> np.ndarray:
    """
    Renders captions on a single video frame.
    This is a core drawing utility.
    """

    # Parse colors
    try:
        bg_color = parse_color(bg_color_str)
        highlight_color = parse_color(highlight_color_str)
        text_color = parse_color(text_color_str) # Keep as RGBA, Pillow handles it
    except ValueError as e:
        print(f"Error parsing colors in render_captions_on_frame: {e}. Using defaults.")
        bg_color = (255, 255, 255, 0)
        highlight_color = (255, 226, 165, 220)
        text_color = (0, 0, 0, 255)

    font_size = max(28, int(video_width * font_size_ratio))
    font = get_font_path(font_path_override, font_size)

    active_display_lines = [] # These are lines from "text_lines" that should be on screen

    # Determine which lines from processed_segments.text_lines are active
    for segment in processed_segments:
        for line_info in segment.get("text_lines", []):
            if line_info["start"] <= current_time < line_info["end"]: # Use < for end to avoid flicker
                active_display_lines.append(line_info)

    if not active_display_lines:
        return frame # No active captions, return original frame

    # Limit to max_lines_on_screen, taking the latest starting ones if too many
    if len(active_display_lines) > max_lines_on_screen:
        active_display_lines.sort(key=lambda x: x['start'], reverse=True)
        active_display_lines = active_display_lines[:max_lines_on_screen]
        active_display_lines.sort(key=lambda x: x['start']) # Restore chronological order for display

    pil_img = cv2_to_pil(frame)
    draw = ImageDraw.Draw(pil_img)

    total_caption_height_estimate = 0
    line_details_for_drawing = []

    for line_info in active_display_lines:
        line_text = line_info["text"]
        try:
            line_bbox = draw.textbbox((0,0), line_text, font=font)
            text_w = line_bbox[2] - line_bbox[0]
            text_h = line_bbox[3] - line_bbox[1]
        except: # Fallback for problematic text/font interaction
            text_w = len(line_text) * (font_size / 2)
            text_h = font_size

        line_details_for_drawing.append({
            "text": line_text, "words": line_info["words"],
            "width": text_w, "height": text_h,
            "start_time": line_info["start"], "end_time": line_info["end"]
        })
        total_caption_height_estimate += text_h + bg_padding * 2 + line_spacing

    # Adjust for last line not needing spacing
    if total_caption_height_estimate > 0:
        total_caption_height_estimate -= line_spacing

    # Position lines from bottom up
    current_y_base = video_height - (int(video_height * bottom_padding_ratio)) - bg_padding

    for i, line_detail in enumerate(reversed(line_details_for_drawing)): # Draw last line first at the bottom
        line_text = line_detail["text"]
        line_words = line_detail["words"]
        text_w, text_h = line_detail["width"], line_detail["height"]

        # Calculate y position for this line's text (anchor is top of text)
        # The background will be drawn around this.
        text_y_position = current_y_base - text_h - ( (len(line_details_for_drawing) -1 -i) * (text_h + bg_padding*2 + line_spacing) if i < len(line_details_for_drawing) -1 else 0)


        # Background for the text line
        bg_x1 = (video_width - text_w) / 2 - bg_padding
        bg_y1 = text_y_position - bg_padding
        bg_x2 = bg_x1 + text_w + bg_padding * 2
        bg_y2 = bg_y1 + text_h + bg_padding * 2

        # Ensure background coordinates are valid before drawing
        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
             draw_rounded_rectangle(draw, (bg_x1, bg_y1, bg_x2, bg_y2), corner_radius, fill=bg_color)
        else:
            print(f"Warning: Invalid background bbox for line: {line_text}")


        # Word-by-word highlighting and drawing
        # Start position for the first word in the line
        current_word_x_offset = (video_width - text_w) / 2

        for word_info in line_words:
            word_text = word_info["text"] # original scripts use word["word"]
            word_start, word_end = word_info["start"], word_info["end"]

            try:
                word_bbox = draw.textbbox((0,0), word_text, font=font)
                word_w = word_bbox[2] - word_bbox[0]
                # word_h = word_bbox[3] - word_bbox[1] # Should be same as text_h for the line
            except:
                word_w = len(word_text) * (font_size / 2)

            is_active_word = word_start <= current_time < word_end

            if is_active_word:
                # Highlight background for the active word
                hl_x1 = current_word_x_offset - bg_padding / 2
                hl_y1 = text_y_position - bg_padding / 2
                hl_x2 = current_word_x_offset + word_w + bg_padding / 2
                hl_y2 = text_y_position + text_h + bg_padding / 2
                if hl_x2 > hl_x1 and hl_y2 > hl_y1:
                    draw_rounded_rectangle(draw, (hl_x1, hl_y1, hl_x2, hl_y2), corner_radius // 2, fill=highlight_color)

            # Draw the word text
            draw.text((current_word_x_offset, text_y_position), word_text, font=font, fill=text_color)

            # Advance x for next word (add space width)
            try:
                space_bbox = draw.textbbox((0,0), " ", font=font)
                space_w = space_bbox[2] - space_bbox[0]
            except:
                space_w = font_size / 3 # Estimate space width

            current_word_x_offset += word_w + space_w

        # Update Y for next line (drawing bottom-up)
        # This logic is implicitly handled by how text_y_position is calculated per line in the loop.
        # current_y_base -= (text_h + bg_padding * 2 + line_spacing)


    return pil_to_cv2(pil_img)

if __name__ == '__main__':
    # Basic test for process_segments_for_caption_lines
    print("Testing caption_utils...")
    sample_segments = [
        {
            "start": 0.0, "end": 5.0, "text": "This is a short test sentence.",
            "words": [
                {"word": "This", "start": 0.0, "end": 0.5, "text": "This"}, {"word": "is", "start": 0.5, "end": 1.0, "text": "is"},
                {"word": "a", "start": 1.0, "end": 1.2, "text": "a"}, {"word": "short", "start": 1.2, "end": 2.0, "text": "short"},
                {"word": "test", "start": 2.0, "end": 2.5, "text": "test"}, {"word": "sentence.", "start": 2.5, "end": 3.0, "text": "sentence."}
            ]
        },
        {   "start": 5.5, "end": 10.0, "text": "Another line here for testing purposes, let's see how it wraps around the screen.",
            "words": [
                {"word": "Another", "start": 5.5, "end": 6.0, "text": "Another"}, {"word": "line", "start": 6.0, "end": 6.3, "text": "line"},
                {"word": "here", "start": 6.3, "end": 6.8, "text": "here"}, {"word": "for", "start": 6.8, "end": 7.0, "text": "for"},
                {"word": "testing", "start": 7.0, "end": 7.5, "text": "testing"}, {"word": "purposes,", "start": 7.5, "end": 8.0, "text": "purposes,"},
                {"word": "let's", "start": 8.0, "end": 8.3, "text": "let's"}, {"word": "see", "start": 8.3, "end": 8.6, "text": "see"},
                {"word": "how", "start": 8.6, "end": 9.0, "text": "how"}, {"word": "it", "start": 9.0, "end": 9.2, "text": "it"},
                {"word": "wraps", "start": 9.2, "end": 9.7, "text": "wraps"}, {"word": "around", "start": 9.7, "end": 10.2, "text": "around"},
                 {"word": "the", "start": 10.2, "end": 10.4, "text": "the"}, {"word": "screen.", "start": 10.4, "end": 11.0, "text": "screen."}
            ] # Ends beyond segment end, should be clipped by calling logic ideally
        }
    ]
    video_w = 1920 # Full HD width
    processed = process_segments_for_caption_lines(sample_segments, video_w, font_size=60)

    for i, seg in enumerate(processed):
        print(f"\nSegment {i+1} (Original: '{seg['text']}'):")
        if "text_lines" in seg:
            for j, line in enumerate(seg["text_lines"]):
                print(f"  Line {j+1} ({line['start']:.2f}s - {line['end']:.2f}s): {line['text']}")
                # print(f"    Words: {line['words']}")
        else:
            print("  No text_lines generated.")

    print("\nBasic caption_utils test complete.")
    # Further testing would involve creating a dummy frame and calling render_captions_on_frame
    # and saving it as an image to visually inspect.
    # Example:
    # dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8) # Black frame
    # rendered_frame = render_captions_on_frame(dummy_frame, 1.5, processed, 1920, 1080)
    # cv2.imwrite("test_caption_frame.png", rendered_frame)
    # print("Test frame 'test_caption_frame.png' saved (if cv2 is available and code uncommented).")
