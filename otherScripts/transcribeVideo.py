import os
import argparse
from pathlib import Path
import sys

# Adjust Python path to include the root directory for core module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.video_utils import extract_audio
from core.transcription_utils import transcribe_audio, set_transcription_device

def transcribe_video_to_text(video_path: str, output_txt_path: Path, model_size: str = "base", device: str = "cpu"):
    """
    Extracts audio from a video, transcribes it using Whisper, and saves the transcription to a text file.
    """
    print(f"Processing video: {video_path}")
    
    # Determine temporary audio file path (e.g., in the same dir as output or a temp dir)
    temp_audio_filename = f"temp_audio_for_{output_txt_path.stem}.wav"
    # Place temp audio in the output directory to handle permissions or relative path issues.
    temp_audio_path = output_txt_path.parent / temp_audio_filename

    set_transcription_device(device)

    extracted_audio_path = extract_audio(video_path, output_path=str(temp_audio_path))

    if not extracted_audio_path or not os.path.exists(extracted_audio_path):
        print(f"Audio extraction failed for {video_path}. Cannot transcribe.")
        return

    try:
        print(f"Transcribing audio from {extracted_audio_path} using model '{model_size}' on device '{device}'...")
        # transcribe_audio from core returns list of segments with text and timestamps.
        # For this script, we just want the concatenated text.
        transcription_segments = transcribe_audio(extracted_audio_path, model_size, word_timestamps=False)
        
        full_transcription_text = "\n".join([segment['text'].strip() for segment in transcription_segments])
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_transcription_text)
        print(f"Transcription saved successfully to: {output_txt_path}")
        
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
    finally:
        # Clean up temporary audio file
        if os.path.exists(extracted_audio_path):
            try:
                os.remove(extracted_audio_path)
                print(f"Cleaned up temporary audio file: {extracted_audio_path}")
            except OSError as e:
                print(f"Warning: Could not remove temporary audio file {extracted_audio_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe video audio to a text file using Whisper.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path for the output .txt file. Defaults to video filename with .txt extension.")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model size to use for transcription (default: base).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Device to use for transcription (e.g., 'cpu', 'gpu'). Default: cpu.")
    
    args = parser.parse_args()

    video_file_path = Path(args.video_path)
    if not video_file_path.is_file():
        print(f"Error: Video file not found at {args.video_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        # Default output path: same name as video, but with .txt extension, in the same directory
        output_path = video_file_path.with_suffix('.txt')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcribe_video_to_text(str(video_file_path), output_path, args.model, args.device)

if __name__ == "__main__":
    main()