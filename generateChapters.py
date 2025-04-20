import os
import subprocess
import numpy as np
import json
import whisper
import argparse
import time
import google.generativeai as genai
from typing import List, Dict, Any

class LLMChapterGenerator:
    """Class to handle LLM API calls for identifying YouTube chapters"""

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

    def generate_chapters(self, transcription_segments, video_duration, min_chapters=5, max_chapters=15):
        """Use LLM to identify logical chapter points from transcription segments"""
        
        # Format the transcription data for the LLM
        transcript_text = ""
        for i, segment in enumerate(transcription_segments):
            start_time = self._format_time(segment["start"])
            end_time = self._format_time(segment["end"])
            transcript_text += f"[{start_time} - {end_time}] {segment['text']}\n"
        
        # Calculate video duration in minutes
        duration_minutes = video_duration / 60
        
        # Create prompt for the LLM
        prompt = f"""
You are a professional YouTube video editor who specializes in creating effective chapter markers.

Below is a transcript of a video that is {duration_minutes:.1f} minutes long. 
The transcript includes timestamps in [hh:mm:ss] format.

TRANSCRIPT:
{transcript_text}

Please generate {min_chapters}-{max_chapters} logical chapter markers for this video by dividing it into meaningful sections.

REQUIREMENTS:
1. First chapter MUST start at 00:00.
2. Chapters should represent natural topical divisions or key moments.
3. Choose concise but descriptive chapter titles (3-7 words each).
4. Ensure time gaps between chapters aren't too short (at least 10+ seconds).
5. Space chapters somewhat evenly throughout the video, but prioritize content transitions.

Format your response as JSON with this structure:
{{
  "chapters": [
    {{
      "time": "00:00",
      "title": "Introduction to Topic"
    }},
    {{
      "time": "01:23",
      "title": "First Main Point"
    }},
    ...
  ]
}}

Ensure ALL timestamps are in the MM:SS or HH:MM:SS format required by YouTube.
"""

        if self.use_gemini:
            return self._call_gemini_api(prompt)
        else:
            return self._fallback_extraction(transcription_segments, video_duration)
            
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
                chapter_data = json.loads(content)
                return chapter_data
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response. Using manual extraction.")
                return self._manually_extract_chapters(content)
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None
            
    def _manually_extract_chapters(self, content):
        """Manually extract chapter information if JSON parsing fails"""
        chapters = []
        
        # Try to find and extract chapter information using regex
        import re
        
        # Look for patterns like timestamps followed by titles
        # This handles different formats like "00:00 - Introduction" or "00:00: Introduction"
        chapter_matches = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|:|–|\|)?\s*([^\n]+)', content)
        
        for time_str, title in chapter_matches:
            chapters.append({
                "time": time_str,
                "title": title.strip()
            })
        
        return {"chapters": chapters}
    
    def _fallback_extraction(self, transcription_segments, video_duration):
        """Simple fallback method if all API calls fail"""
        chapters = []
        
        # Always include intro at 00:00
        chapters.append({
            "time": "00:00",
            "title": "Introduction"
        })
        
        # Create evenly spaced chapters
        num_chapters = 5  # Default to 5 chapters minimum
        segment_duration = video_duration / num_chapters
        
        for i in range(1, num_chapters):
            chapter_time = i * segment_duration
            
            # Find the closest transcript segment to this time
            closest_segment = None
            min_time_diff = float('inf')
            
            for segment in transcription_segments:
                time_diff = abs(segment["start"] - chapter_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_segment = segment
            
            # Generate a simple title from the segment text
            if closest_segment:
                title_text = closest_segment["text"]
                if len(title_text) > 40:
                    title_text = title_text[:37] + "..."
                
                chapters.append({
                    "time": self._format_time(closest_segment["start"]),
                    "title": title_text
                })
        
        return {"chapters": chapters}
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS format for YouTube"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
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
    result = model.transcribe(audio_path)
    
    # Extract segments from the result
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    return segments


def get_video_duration(video_path):
    """Get the duration of the video in seconds"""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return float(output)


def review_chapters(chapters):
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
    parser.add_argument("--output-dir", default="youtube_chapters", help="Directory to save output files")
    parser.add_argument("--min-chapters", type=int, default=5, help="Minimum number of chapters to generate")
    parser.add_argument("--max-chapters", type=int, default=15, help="Maximum number of chapters to generate")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--api-key", help="API key for LLM service (optional)")
    parser.add_argument("--no-review", action="store_true", help="Skip chapter review")
    parser.add_argument("--add-chapter", action="store_true", help="Add a custom chapter")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Step 1: Get video duration
    print("Getting video duration...")
    video_duration = get_video_duration(args.video_path)
    print(f"Video duration: {video_duration} seconds")
    
    # Step 2: Extract audio from video
    print("Extracting audio from video...")
    audio_path = extract_audio(args.video_path)
    
    # Step 3: Transcribe audio
    print("Transcribing audio...")
    transcription_segments = transcribe_audio(audio_path, args.whisper_model)
    
    # Save transcription to file
    transcription_path = os.path.join(args.output_dir, "transcription.json")
    with open(transcription_path, "w") as f:
        json.dump(transcription_segments, f, indent=2)
    
    print(f"Transcription saved to {transcription_path}")
    
    # Step 4: Generate chapters using LLM
    print("Generating chapters using LLM...")
    chapter_generator = LLMChapterGenerator(api_key=args.api_key)
    chapter_suggestions = chapter_generator.generate_chapters(
        transcription_segments, 
        video_duration,
        min_chapters=args.min_chapters, 
        max_chapters=args.max_chapters
    )
    
    if not chapter_suggestions or "chapters" not in chapter_suggestions or not chapter_suggestions["chapters"]:
        print("No chapters generated. Exiting.")
        os.remove(audio_path)
        return
    
    chapters = chapter_suggestions["chapters"]
    print(f"Generated {len(chapters)} chapter markers")
    
    # Save chapter suggestions to file
    suggestions_path = os.path.join(args.output_dir, "chapter_suggestions.json")
    with open(suggestions_path, "w") as f:
        json.dump(chapter_suggestions, f, indent=2)
    
    print(f"Chapter suggestions saved to {suggestions_path}")
    
    # Step 5: Review chapters if requested
    if not args.no_review:
        print("\nReviewing chapters...")
        
        while True:
            reviewed_chapters = review_chapters(chapters)
            
            # Ask if user wants to add a custom chapter
            add_custom = input("\nWould you like to add a custom chapter? (y/n): ").lower()
            if add_custom == 'y':
                chapter_time = input("Enter chapter time (MM:SS or HH:MM:SS): ")
                chapter_title = input("Enter chapter title: ")
                chapters = add_chapter(reviewed_chapters, chapter_time, chapter_title)
                continue
            else:
                chapters = reviewed_chapters
                break
    
    # Step 6: Format chapters for YouTube
    youtube_chapters = format_chapters_for_youtube(chapters)
    
    # Save YouTube-formatted chapters
    youtube_path = os.path.join(args.output_dir, "youtube_chapters.txt")
    with open(youtube_path, "w") as f:
        f.write(youtube_chapters)
    
    print(f"\nYouTube chapters saved to {youtube_path}")
    print("\nCopy and paste these into your YouTube description:")
    print("-" * 50)
    print(youtube_chapters)
    print("-" * 50)
    
    # Clean up
    os.remove(audio_path)
    print("\nProcess complete!")


if __name__ == "__main__":
    main()  