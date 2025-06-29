import whisper
import os
import torch # For CUDA check
from typing import List, Dict, Any

# Global variable to store the chosen device for transcription
_transcription_device = "cpu"

def set_transcription_device(requested_device: str):
    """
    Sets the device for Whisper transcription (cpu or cuda).
    Validates GPU availability if 'gpu' or 'cuda' is requested.
    """
    global _transcription_device
    requested_device = requested_device.lower()

    if requested_device in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            _transcription_device = "cuda"
            print("Transcription device set to: CUDA (GPU)")
        else:
            _transcription_device = "cpu"
            print("Warning: GPU (CUDA) requested, but CUDA is not available. Falling back to CPU.")
    elif requested_device == "cpu":
        _transcription_device = "cpu"
        print("Transcription device set to: CPU")
    else:
        print(f"Warning: Invalid device '{requested_device}' specified. Using default CPU.")
        _transcription_device = "cpu" # Default to CPU if invalid choice

def get_transcription_device() -> str:
    """Returns the currently configured transcription device."""
    return _transcription_device

def transcribe_audio(audio_path: str, whisper_model_size: str = "base", word_timestamps: bool = False) -> List[Dict[str, Any]]:
    """
    Transcribe audio using Whisper and return segments.
    Includes word timestamps if word_timestamps is True.
    Uses the device set by set_transcription_device().
    """
    current_device = get_transcription_device()
    print(f"Loading Whisper model '{whisper_model_size}' onto device '{current_device}'...")

    try:
        model = whisper.load_model(whisper_model_size, device=current_device)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        # Fallback or re-raise depending on desired strictness
        if current_device == "cuda":
            print("Attempting to load model on CPU as a fallback...")
            try:
                set_transcription_device("cpu") # Update global state
                current_device = get_transcription_device()
                model = whisper.load_model(whisper_model_size, device=current_device)
                print(f"Successfully loaded model on CPU after '{e}' on GPU.")
            except Exception as fallback_e:
                print(f"Fallback to CPU also failed: {fallback_e}")
                raise fallback_e # Re-raise if CPU fallback also fails
        else:
            raise e # Re-raise if it was already CPU or another error

    print(f"Transcribing audio file: {audio_path} (Word timestamps: {word_timestamps}) using device: {current_device}")
    result = model.transcribe(audio_path, word_timestamps=word_timestamps)

    segments = []
    for segment_data in result["segments"]:
        segment = {
            "start": segment_data["start"],
            "end": segment_data["end"],
            "text": segment_data["text"],
        }
        if word_timestamps and "words" in segment_data:
            segment["words"] = [{
                "word": word_data["word"],
                "start": word_data["start"],
                "end": word_data["end"],
                # Whisper might also provide 'confidence' or other metrics per word
            } for word_data in segment_data["words"]]
        else:
            # Ensure 'words' key exists even if empty, for consistent data structure
            segment["words"] = []
        segments.append(segment)

    return segments

def review_transcription(transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Allow user to review and edit the transcription segments."""
    print("\n=== Transcription Review ===")
    print("Review each segment and correct any errors.")
    print("For each segment, you can:")
    print("  [enter] - Accept as is")
    print("  [edit text] - Type corrected text")
    print("  q - Finish review and save changes")
    print("  s - Skip to end without further review\n")

    updated_segments = []
    for i, segment in enumerate(transcription_segments):
        print(f"\nSegment {i+1}/{len(transcription_segments)}")
        # Assuming format_time will be available or we add it to a utils module
        try:
            # Attempt to use format_time if it's defined elsewhere (e.g. in a general utils later)
            print(f"[{format_time(segment['start'])} - {format_time(segment['end'])}]")
        except NameError:
             print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]") # Fallback timing format

        print(f"Current text: {segment['text']}")

        user_input = input("Correction (or enter to accept): ")

        if user_input.lower() == 'q':
            print("Finishing review...")
            updated_segments.extend(transcription_segments[i:]) # Add remaining unreviewed segments
            break
        elif user_input.lower() == 's':
            print("Skipping remaining segments...")
            updated_segments.extend(transcription_segments[i:]) # Add remaining unreviewed segments
            break
        elif user_input:
            segment['text'] = user_input
            print(f"Updated: {user_input}")

            # Basic word re-estimation if text is changed.
            # This is a simplification; more advanced logic might be needed for precise word timings.
            if segment.get('words'): # Check if 'words' key exists
                words = user_input.split()
                num_words = len(words)
                if num_words > 0:
                    segment_duration = segment['end'] - segment['start']
                    avg_word_duration = segment_duration / num_words
                    new_words_data = []
                    current_time = segment['start']
                    for word_text in words:
                        word_end_time = current_time + avg_word_duration
                        new_words_data.append({
                            "word": word_text,
                            "start": current_time,
                            "end": min(word_end_time, segment['end']) # Ensure word end doesn't exceed segment end
                        })
                        current_time = word_end_time
                    segment['words'] = new_words_data
                else:
                    segment['words'] = [] # No words if text is empty

        updated_segments.append(segment)

    print("\nTranscription review complete!")
    return updated_segments

# Helper function for formatting time, can be moved to a more general util if needed
def format_time(seconds: float) -> str:
    """Format seconds to mm:ss format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
