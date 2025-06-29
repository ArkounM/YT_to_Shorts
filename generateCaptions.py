import os
import subprocess
import cv2
import json
import argparse
from typing import List, Dict, Any
from collections import deque

# Core utilities
from core.video_utils import extract_audio, get_video_dimensions
from core.transcription_utils import transcribe_audio, review_transcription, set_transcription_device, format_time
from core.caption_utils import (
    process_segments_for_caption_lines,
    render_captions_on_frame,
    parse_color as parse_color_util # Renamed to avoid conflict if main defines its own
)


def create_blurred_background(frame, x1, y1, x2, y2, blur_amount=15):
    """Create a blurred background from a region of the frame"""
    # Extract the region to blur
    region = frame[y1:y2, x1:x2]
    if region.size == 0: # check if region is empty
        return frame # return original frame if region is empty
    
    # Apply blur
    blurred_region = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
    
    # Create a copy of the frame to modify
    output_frame = frame.copy()
    output_frame[y1:y2, x1:x2] = blurred_region
    return output_frame


def caption_video(
    video_path: str,
    output_path: str,
    processed_segments: List[Dict[str, Any]],
    font_path: str, # Added font_path argument
    bg_color_str: str = "255,255,255,0",
    highlight_color_str: str = "255,226,165,220",
    text_color_str: str = "0,0,0,255" # Added alpha to text_color
    ):
    """
    Add captions to the entire video using the refactored caption rendering utility.
    processed_segments should already have 'text_lines' from process_segments_for_caption_lines.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video_output_path = f"{output_path}_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 codec
    out_video_writer = cv2.VideoWriter(temp_video_output_path, fourcc, fps, (width, height))

    if not out_video_writer.isOpened():
        print("Failed to open video writer with H.264 (avc1). Trying MPEG-4 (mp4v)...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_writer = cv2.VideoWriter(temp_video_output_path, fourcc, fps, (width, height))
        if not out_video_writer.isOpened():
            print("Error: Could not open video writer with either H.264 or MPEG-4 codec.")
            cap.release()
            return None

    print(f"Processing {total_frames} frames at {fps:.2f} fps for video: {video_path}")
    
    # For stateful animation of word highlights
    last_highlighted_words_deque = deque(maxlen=5)
    
    # Variables for managing caption visibility during silence
    last_text_end_time = 0.0
    silence_duration_to_hide_captions = 2.0 # seconds

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx + 1}/{total_frames}. Stopping.")
            break

        current_time_sec = frame_idx / fps

        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

        # Determine if captions should be shown based on current_time_sec and segment timings
        # This logic is now mostly inside render_captions_on_frame, but we can add silence handling here
        
        # Simplified active_lines check for silence detection (can be more sophisticated)
        current_active_lines = []
        for segment in processed_segments:
            for line in segment.get("text_lines", []):
                if line["start"] <= current_time_sec < line["end"]:
                    current_active_lines.append(line)
                    break # Found an active line in this segment
            if current_active_lines:
                last_text_end_time = max(last_text_end_time, current_active_lines[-1]["end"])
                break
        
        show_captions_for_current_frame = True
        if not current_active_lines: # No lines currently active
            if current_time_sec - last_text_end_time >= silence_duration_to_hide_captions:
                show_captions_for_current_frame = False
            else:
                # In the brief period after text ends but before silence_duration,
                # we might want to keep the last said line on screen.
                # render_captions_on_frame can handle this by checking segment end times.
                pass


        if show_captions_for_current_frame:
            # The original script had a blur background effect tied to caption_bg_bbox.
            # This needs to be integrated carefully if desired, or simplified.
            # For now, render_captions_on_frame handles its own background.
            # If a full-frame or region blur is needed *behind* the caption box, it would be done here.
            # Example: frame = create_blurred_background(frame, x1,y1,x2,y2) if conditions met.
            # This part is complex from the original script and needs careful porting if the exact effect is required.
            # The original script blurred a region *behind* the caption text.
            # render_captions_on_frame draws its own rounded rectangle background.
            # For simplicity, we'll rely on render_captions_on_frame's own background for now.

            frame_with_captions = render_captions_on_frame(
                frame=frame,
                current_time=current_time_sec,
                processed_segments=processed_segments,
                video_width=width,
                video_height=height,
                font_path_override=font_path, # Pass font_path
                bg_color_str=bg_color_str,
                highlight_color_str=highlight_color_str,
                text_color_str=text_color_str,
                last_highlighted_words_deque=last_highlighted_words_deque
            )
            out_video_writer.write(frame_with_captions)
        else:
            out_video_writer.write(frame) # Write original frame if no captions

    cap.release()
    out_video_writer.release()
    print(f"Finished processing frames. Video with captions (no audio) saved to: {temp_video_output_path}")

    # Combine video with original audio
    final_output_video_path = f"{output_path}"
    # Check if original video had audio. If not, -map 1:a:0 might fail.
    # A more robust way is to check audio streams with ffprobe first.
    # For now, assume audio exists or ffmpeg handles it gracefully.
    combine_command = (
        f"ffmpeg -i \"{temp_video_output_path}\" -i \"{video_path}\" "
        f"-c:v copy -map 0:v:0 -map 1:a:0? -shortest \"{final_output_video_path}\" -y"
    )
    # Using -map 1:a:0? makes the audio mapping optional; if no audio stream in video_path, it won't error.

    print(f"Adding audio to video: {combine_command}")
    try:
        subprocess.run(combine_command, shell=True, check=True, capture_output=True, text=True)
        print(f"Successfully combined audio and video into: {final_output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining audio and video: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        # Fallback: if audio merge fails, rename the video-only file.
        if os.path.exists(temp_video_output_path):
            os.rename(temp_video_output_path, final_output_video_path)
            print(f"Warning: Audio merge failed. Output video (no audio from original): {final_output_video_path}")
            return final_output_video_path # Return path to video-only file
        return None # Failure

    # Clean up temporary video file
    if os.path.exists(temp_video_output_path):
        try:
            os.remove(temp_video_output_path)
            print(f"Cleaned up temporary file: {temp_video_output_path}")
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_video_output_path}: {e}")
            
    if os.path.exists(final_output_video_path) and os.path.getsize(final_output_video_path) > 0:
        return final_output_video_path
    else:
        print(f"Error: Final output video file not found or is empty: {final_output_video_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Caption an entire video with word-level subtitles")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-path", help="Path for the output video file", default="captioned_video.mp4")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Device to use for Whisper transcription (cpu or gpu)")
    parser.add_argument("--transcription-path", help="Path to existing transcription JSON (optional)")
    parser.add_argument("--skip-review", action="store_true", help="Skip transcription review")
    
    # Add color customization options
    parser.add_argument("--font-path", default=None, help="Path to .ttf or .otf font file for captions")
    parser.add_argument("--bg-color", default="255,255,255,0", 
                        help="Background color for captions in format 'r,g,b,a' (e.g., '255,255,255,0')")
    parser.add_argument("--highlight-color", default="255,226,165,220", 
                        help="Highlight color for active words in format 'r,g,b,a' (e.g., '255,226,165,220')")
    parser.add_argument("--text-color", default="0,0,0,255",
                        help="Text color for captions in format 'r,g,b,a' (e.g., '0,0,0,255')")
    
    args = parser.parse_args()

    # Set transcription device
    set_transcription_device(args.device) # Will be enhanced later for actual GPU check

    # Validate colors (basic validation, detailed parsing is in caption_utils and render_captions_on_frame)
    try:
        parse_color_util(args.bg_color)
        parse_color_util(args.highlight_color)
        parse_color_util(args.text_color)
    except ValueError as e:
        print(f"Error parsing color argument: {e}")
        return
    
    # Create output directory if needed
    output_dir_path = os.path.dirname(args.output_path)
    if output_dir_path and not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created output directory: {output_dir_path}")
    
    # Load existing transcription or create new one
    if args.transcription_path and os.path.exists(args.transcription_path):
        print(f"Loading existing transcription from {args.transcription_path}")
        with open(args.transcription_path, "r", encoding="utf-8") as f:
            transcription_segments = json.load(f)
    else:
        # Extract audio from video
        print(f"Extracting audio from video: {args.video_path}...")
        temp_audio_path = os.path.join(output_dir_path or ".", "temp_audio.wav")
        audio_path = extract_audio(args.video_path, output_path=temp_audio_path)
        
        if not audio_path:
            print("Audio extraction failed. Exiting.")
            return

        # Transcribe audio
        print(f"Transcribing audio using Whisper model '{args.whisper_model}'...")
        # Pass word_timestamps=True as it's needed for detailed captioning
        transcription_segments = transcribe_audio(audio_path, args.whisper_model, word_timestamps=True)
        
        # Save original transcription to file
        original_transcription_filename = os.path.splitext(os.path.basename(args.output_path))[0] + "_original_transcription.json"
        original_transcription_path = os.path.join(output_dir_path or ".", original_transcription_filename)
        
        with open(original_transcription_path, "w", encoding="utf-8") as f:
            json.dump(transcription_segments, f, indent=2, ensure_ascii=False)
        print(f"Original transcription saved to {original_transcription_path}")
        
        # Clean up audio file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
            except OSError as e:
                print(f"Warning: Could not remove temporary audio file {audio_path}: {e}")

    if not transcription_segments:
        print("Transcription result is empty. Cannot proceed with captioning. Exiting.")
        return

    # Review transcription if not skipped
    if not args.skip_review:
        print("\nStarting transcription review process...")
        transcription_segments = review_transcription(transcription_segments) # Uses format_time from transcription_utils
        
        # Save reviewed transcription
        reviewed_transcription_filename = os.path.splitext(os.path.basename(args.output_path))[0] + "_reviewed_transcription.json"
        reviewed_transcription_path = os.path.join(output_dir_path or ".", reviewed_transcription_filename)
        with open(reviewed_transcription_path, "w", encoding="utf-8") as f:
            json.dump(transcription_segments, f, indent=2, ensure_ascii=False)
        print(f"Reviewed transcription saved to {reviewed_transcription_path}")
    
    # Get video info to calculate dimensions for captions
    video_width, _ = get_video_dimensions(args.video_path)
    if video_width is None:
        print("Could not get video dimensions. Exiting.")
        return
    
    # Process segments for creating display lines for captions
    print("Processing segments for caption line generation...")
    # Pass font_path and desired font_size for accurate line breaking
    processed_segments_for_lines = process_segments_for_caption_lines(
        segments=transcription_segments,
        video_width=video_width,
        font_path=args.font_path, # Allow user to specify font for line calculation
        # font_size can also be passed if we want to override default calculation in process_segments
    )
    
    # Caption the video
    print("Adding captions to video...")
    captioned_video_path = caption_video(
        video_path=args.video_path,
        output_path=args.output_path,
        processed_segments=processed_segments_for_lines,
        font_path=args.font_path, # Pass font_path to caption_video for rendering
        bg_color_str=args.bg_color,
        highlight_color_str=args.highlight_color,
        text_color_str=args.text_color
    )
    
    if captioned_video_path:
        print(f"\nProcess complete! Captioned video saved to {captioned_video_path}")
        print(f"Caption settings used:")
        print(f"  Font path: {args.font_path or 'Auto-selected'}")
        print(f"  Background color: {args.bg_color}")
        print(f"  Highlight color: {args.highlight_color}")
        print(f"  Text color: {args.text_color}")
    else:
        print("\nFailed to create captioned video.")


if __name__ == "__main__":
    main()