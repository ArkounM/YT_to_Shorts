import os
import json
import argparse
import time # For sorting chapters
from typing import List, Dict, Any, Optional

# Core utilities
from core.video_utils import extract_audio, get_video_duration
from core.transcription_utils import transcribe_audio, set_transcription_device
from core.llm_utils import get_llm_provider, LLMProvider # Using the factory and base class


class ChapterGenerator:
    """Handles chapter generation using a specified LLM provider."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def _format_time_for_youtube(self, seconds: float) -> str:
        """Formats seconds to MM:SS or HH:MM:SS for YouTube chapters."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def generate_chapters_with_llm(self,
                                   transcription_segments: List[Dict[str, Any]],
                                   video_duration_seconds: float,
                                   min_chapters: int = 5,
                                   max_chapters: int = 15) -> Optional[Dict[str, Any]]:
        """Uses LLM to identify logical chapter points from transcription segments."""
        
        transcript_text_for_llm = ""
        for segment in transcription_segments:
            # Use the LLM provider's time formatting utility if available, or default
            start_time_str = self.llm_provider._format_time_for_llm(segment["start"]) if hasattr(self.llm_provider, '_format_time_for_llm') else self._format_time_for_youtube(segment["start"])
            # end_time_str = self.llm_provider._format_time_for_llm(segment["end"]) if hasattr(self.llm_provider, '_format_time_for_llm') else self._format_time_for_youtube(segment["end"])
            # transcript_text_for_llm += f"[{start_time_str} - {end_time_str}] {segment['text']}\n"
            transcript_text_for_llm += f"[{start_time_str}] {segment['text']}\n"


        duration_minutes = video_duration_seconds / 60
        
        prompt = f"""
You are a professional YouTube video editor specializing in creating effective chapter markers.
The video is {duration_minutes:.1f} minutes long.
Below is the transcript with timestamps in [HH:MM:SS] or [MM:SS] format.

TRANSCRIPT:
{transcript_text_for_llm}

TASK:
Generate between {min_chapters} and {max_chapters} logical chapter markers for this video.

REQUIREMENTS:
1. The first chapter MUST start at "00:00" (or "0:00:00").
2. Chapters should mark significant topical shifts or key moments.
3. Titles should be concise and descriptive (3-7 words).
4. Ensure reasonable time gaps between chapters (e.g., at least 15-30 seconds, unless content dictates closer markers for very distinct short sections).
5. Prioritize natural content transitions over strict even spacing.

OUTPUT FORMAT:
Return your response as a JSON object with a single key "chapters".
The value of "chapters" should be a list of objects, each with "time" (string formatted as MM:SS or HH:MM:SS) and "title" (string).

Example:
{{
  "chapters": [
    {{ "time": "00:00", "title": "Introduction to the Topic" }},
    {{ "time": "02:35", "title": "Exploring the First Key Point" }},
    {{ "time": "05:10", "title": "Practical Demonstration" }}
  ]
}}
Ensure all timestamps are correctly formatted for YouTube.
"""
        print("Sending chapter generation prompt to LLM...")
        # Use the generate_json method from the LLM provider
        chapter_data = self.llm_provider.generate_json(prompt)

        if chapter_data and "chapters" in chapter_data and isinstance(chapter_data["chapters"], list):
            # Basic validation of chapter format
            valid_chapters = []
            for ch in chapter_data["chapters"]:
                if isinstance(ch, dict) and "time" in ch and "title" in ch:
                    valid_chapters.append(ch)
                else:
                    print(f"Warning: Skipping malformed chapter data from LLM: {ch}")
            if not valid_chapters and chapter_data["chapters"]: # LLM returned chapters but all were malformed
                 print("Warning: LLM returned chapter data, but none were in the expected format.")
                 return self._fallback_chapter_extraction(transcription_segments, video_duration_seconds, num_chapters_target=min_chapters)
            elif not valid_chapters: # LLM returned no chapters or malformed structure
                 print("LLM did not return valid chapter data. Attempting fallback.")
                 return self._fallback_chapter_extraction(transcription_segments, video_duration_seconds, num_chapters_target=min_chapters)

            chapter_data["chapters"] = valid_chapters
            return chapter_data
        else:
            print("LLM response for chapters was not in the expected JSON format or missing 'chapters' list. Attempting fallback.")
            # Fallback if LLM fails to produce valid JSON or expected structure
            return self._fallback_chapter_extraction(transcription_segments, video_duration_seconds, num_chapters_target=min_chapters)

    def _fallback_chapter_extraction(self,
                                     transcription_segments: List[Dict[str, Any]],
                                     video_duration_seconds: float,
                                     num_chapters_target: int = 5) -> Dict[str, List[Dict[str, str]]]:
        """Generates evenly spaced basic chapters as a fallback."""
        print("Using fallback chapter generation method (evenly spaced).")
        chapters = [{"time": "00:00", "title": "Introduction"}]
        if video_duration_seconds <= 0 or not transcription_segments:
            return {"chapters": chapters}

        # Ensure at least 2 chapters if duration allows
        num_chapters_to_generate = max(2, num_chapters_target)

        # Avoid division by zero if num_chapters_to_generate is 1 (only intro)
        if num_chapters_to_generate <= 1:
             return {"chapters": chapters}

        # Calculate interval, ensuring we don't try to make chapters too close to start/end
        # Effective duration for placing chapters (not too close to start/end)
        effective_duration = video_duration_seconds * 0.9
        start_offset = video_duration_seconds * 0.05

        # Interval between the additional chapters (num_chapters_to_generate - 1 segments)
        chapter_interval = effective_duration / (num_chapters_to_generate -1)


        for i in range(1, num_chapters_to_generate):
            # Calculate time for this chapter
            # Ensure chapter_time is not before the first segment's start or too close to video end
            chapter_time_seconds = start_offset + (i-1) * chapter_interval
            chapter_time_seconds = min(chapter_time_seconds, video_duration_seconds * 0.95) # Cap at 95% of duration

            # Find the closest segment text for a title (simple approach)
            title = f"Chapter {i + 1}" # Default title
            closest_segment_text = None
            min_diff = float('inf')

            for seg in transcription_segments:
                diff = abs(seg['start'] - chapter_time_seconds)
                if diff < min_diff:
                    min_diff = diff
                    # Try to get a few words for the title
                    closest_segment_text = " ".join(seg['text'].split()[:5])

            if closest_segment_text:
                title = closest_segment_text + ("..." if len(seg['text'].split()) > 5 else "")

            chapters.append({
                "time": self._format_time_for_youtube(chapter_time_seconds),
                "title": title
            })

        # Sort chapters by time just in case, and remove duplicates by time
        unique_chapters = []
        seen_times = set()
        chapters.sort(key=lambda x: time_to_seconds(x["time"]))
        for ch in chapters:
            if ch["time"] not in seen_times:
                unique_chapters.append(ch)
                seen_times.add(ch["time"])

        return {"chapters": unique_chapters}


def review_chapters(chapters: List[Dict[str,str]]):
    """Allow user to review and edit chapters before finalizing"""
    reviewed_chapters = []
    
    print("\n=== Chapter Markers to Review ===")
    
    for i, chapter in enumerate(chapters):
        print(f"\nChapter {i+1}:")
        
        while True:  # Continue until user decides to approve or skip
            # Display current chapter info
            print(f"  Time: {chapter['time']}")
            print(f"  Title: {chapter['title']}")
            
            # Ask for user action
            action = input("\nActions: [a]pprove, [e]dit, [s]kip, [q]uit review: ").lower()
            
            if action == 'a':
                reviewed_chapters.append(chapter)
                print("Chapter approved!")
                break
                
            elif action == 'e':
                # Edit chapter
                new_time = input(f"New time (current: {chapter['time']}, format MM:SS or HH:MM:SS): ")
                if new_time:
                    chapter["time"] = new_time
                
                new_title = input(f"New title (current: {chapter['title']}): ")
                if new_title:
                    chapter["title"] = new_title
                
                print(f"Chapter updated: {chapter['time']} - {chapter['title']}")
                
            elif action == 's':
                print("Chapter skipped.")
                break
                
            elif action == 'q':
                print("Review completed.")
                return reviewed_chapters
                
            else:
                print("Invalid action. Please try again.")
    
    return reviewed_chapters


def format_chapters_for_youtube(chapters):
    """Format chapters in YouTube-ready format"""
    if not chapters:
        return "No chapters generated."
    
    # Ensure the first chapter starts at 00:00
    has_zero_start = False
    for chapter in chapters:
        if chapter["time"] == "00:00":
            has_zero_start = True
            break
    
    if not has_zero_start:
        chapters.insert(0, {"time": "00:00", "title": "Introduction"})
    
    # Sort chapters by time
    chapters.sort(key=lambda x: time_to_seconds(x["time"]))
    
    # Format for YouTube description
    youtube_format = ""
    for chapter in chapters:
        youtube_format += f"{chapter['time']} {chapter['title']}\n"
    
    return youtube_format


def time_to_seconds(time_str):
    """Convert time string (MM:SS or HH:MM:SS) to seconds for sorting"""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        return 0


def add_chapter(chapters, time, title):
    """Add a new chapter at specified time with given title"""
    chapters.append({"time": time, "title": title})
    return chapters


def main():
    parser = argparse.ArgumentParser(description="Generate YouTube chapters from video transcript")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default="youtube_chapters", help="Directory to save output files.")
    parser.add_argument("--min-chapters", type=int, default=5, help="Minimum number of chapters to generate.")
    parser.add_argument("--max-chapters", type=int, default=15, help="Maximum number of chapters to generate.")

    # Transcription arguments
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for transcription.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Device for Whisper transcription (cpu or gpu).")
    parser.add_argument("--transcription-path", help="Path to existing transcription JSON file (optional).")

    # LLM arguments
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "openai", "openrouter", "deepseek"],
                        help="LLM provider to use for chapter generation.")
    parser.add_argument("--llm-model", default=None, help="Specific LLM model name to use (optional, defaults to provider's default).")
    parser.add_argument("--api-key", help="API key for the selected LLM service (reads from environment if not set).")

    parser.add_argument("--no-review", action="store_true", help="Skip interactive chapter review.")
    # The --add-chapter during review is handled by the review_chapters function itself.
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- 1. Transcription ---
    set_transcription_device(args.device)
    transcription_segments = None

    if args.transcription_path and os.path.exists(args.transcription_path):
        print(f"Loading existing transcription from: {args.transcription_path}")
        with open(args.transcription_path, "r", encoding="utf-8") as f:
            transcription_segments = json.load(f)
    else:
        print(f"Starting transcription process for: {args.video_path}")
        temp_audio_output_path = os.path.join(args.output_dir, "temp_chapters_audio.wav")
        audio_path = extract_audio(args.video_path, output_path=temp_audio_output_path)
        if not audio_path:
            print("Audio extraction failed. Cannot generate chapters.")
            return

        print(f"Transcribing audio with Whisper model: {args.whisper_model} on device: {args.device}")
        # For chapter generation, word timestamps are not strictly necessary but don't hurt.
        # Using word_timestamps=False for slightly faster transcription if segments are sufficient.
        transcription_segments = transcribe_audio(audio_path, args.whisper_model, word_timestamps=False)

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
            except OSError as e:
                 print(f"Warning: Could not delete temporary audio file {audio_path}: {e}")

        # Save the new transcription
        new_transcription_file = os.path.join(args.output_dir, "transcription_for_chapters.json")
        with open(new_transcription_file, "w", encoding="utf-8") as f:
            json.dump(transcription_segments, f, indent=2, ensure_ascii=False)
        print(f"Transcription saved to: {new_transcription_file}")

    if not transcription_segments:
        print("No transcription data available. Cannot generate chapters.")
        return

    # --- 2. Get Video Duration ---
    print("Fetching video duration...")
    video_duration_secs = get_video_duration(args.video_path)
    if video_duration_secs is None:
        print("Could not determine video duration. Cannot accurately generate chapters.")
        # We could attempt to proceed without it, but LLM prompt benefits from it.
        return
    print(f"Video duration: {video_duration_secs:.2f} seconds")

    # --- 3. Initialize LLM and Generate Chapters ---
    print(f"Initializing LLM provider: {args.llm_provider} (Model: {args.llm_model or 'default'})")
    try:
        llm_service = get_llm_provider(args.llm_provider, api_key=args.api_key, model=args.llm_model)
    except (ValueError, ImportError, ConnectionError) as e:
        print(f"Error initializing LLM provider: {e}")
        return

    chapter_generator_instance = ChapterGenerator(llm_service)
    print("Generating chapter suggestions with LLM...")
    chapter_suggestions_data = chapter_generator_instance.generate_chapters_with_llm(
        transcription_segments,
        video_duration_secs,
        min_chapters=args.min_chapters,
        max_chapters=args.max_chapters
    )

    if not chapter_suggestions_data or "chapters" not in chapter_suggestions_data or not chapter_suggestions_data["chapters"]:
        print("Failed to generate chapter suggestions from LLM or fallback. Exiting.")
        return
    
    generated_chapters = chapter_suggestions_data["chapters"]
    print(f"LLM (or fallback) generated {len(generated_chapters)} chapter markers.")

    # Save raw suggestions
    raw_suggestions_path = os.path.join(args.output_dir, "chapter_suggestions_raw.json")
    with open(raw_suggestions_path, "w", encoding="utf-8") as f:
        json.dump(chapter_suggestions_data, f, indent=2, ensure_ascii=False)
    print(f"Raw chapter suggestions saved to: {raw_suggestions_path}")

    # --- 4. Review Chapters (Optional) ---
    final_chapters = generated_chapters
    if not args.no_review:
        print("\nStarting chapter review process...")
        # The review_chapters function now handles adding custom chapters internally.
        final_chapters = review_chapters(generated_chapters)
    
    if not final_chapters:
        print("No chapters remaining after review. Exiting.")
        return

    # --- 5. Format and Save Final Chapters ---
    youtube_formatted_chapters = format_chapters_for_youtube(final_chapters)
    
    output_youtube_chapters_txt = os.path.join(args.output_dir, "youtube_chapters_formatted.txt")
    with open(output_youtube_chapters_txt, "w", encoding="utf-8") as f:
        f.write(youtube_formatted_chapters)
    print(f"\nFormatted YouTube chapters saved to: {output_youtube_chapters_txt}")
    
    final_chapters_json_path = os.path.join(args.output_dir, "chapters_final.json")
    with open(final_chapters_json_path, "w", encoding="utf-8") as f:
        json.dump({"chapters": final_chapters}, f, indent=2, ensure_ascii=False)
    print(f"Final chapter data (JSON) saved to: {final_chapters_json_path}")

    print("\n--- YouTube Chapter List ---")
    print(youtube_formatted_chapters)
    print("----------------------------")
    print("Process complete!")


if __name__ == "__main__":
    main()  