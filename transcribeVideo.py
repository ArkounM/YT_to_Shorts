import whisper
import subprocess
import os
import argparse
from pathlib import Path

def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file using ffmpeg"""
    print(f"Extracting audio from: {video_path}")
    command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{output_path}" -y'
    subprocess.call(command, shell=True)
    return output_path

def transcribe_video(video_path, output_path=None, model_size="base"):
    """Transcribe video using Whisper and save to text file"""
    
    # Handle output path
    if output_path is None:
        output_path = Path(video_path).with_suffix('.txt')
    
    # Extract audio first
    audio_path = extract_audio(video_path)
    
    try:
        # Load Whisper model
        print(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        
        # Transcribe audio
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        
        # Write transcription to file
        print(f"Writing transcription to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        print("Transcription complete!")
        
    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("Cleaned up temporary audio file")

def main():
    parser = argparse.ArgumentParser(description="Transcribe video audio to text file")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", help="Output text file path (optional)")
    parser.add_argument("--model", default="base", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model size (default: base)")
    
    args = parser.parse_args()
    transcribe_video(args.video_path, args.output, args.model)

if __name__ == "__main__":
    main()