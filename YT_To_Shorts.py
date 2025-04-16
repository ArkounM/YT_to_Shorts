import whisper
from moviepy.editor import VideoFileClip, compositeVideoClip, TextClip, vfx
import os

# Step 1 - Transcribe the video using Whisper
def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["segments"]

# Step 2 - Select Segment for shorts 
def select_segment(segments):
    for seg in segments:
        if 10 <= (seg["end"] - seg["start"]) <= 60:
            return seg
    return segments[0] #fallback

# Step 3 - format for vertical video with blurred background
def format_vertical_with_blur(clip):
    # Resize for vertical fram
    foreground = clip.resize(width=1080)

    # Create blurred background
    background = (
        clip.resize(height=1920)
            .crop(width=1080, x_center=clip.w / 2)
            .fx(vfx.blur, 30)
    )

    # Combine Layers
    final = compositeVideoClip([background, foreground.set_position("center")])
    return final

# Step 4 - Add Word-by-Word Captions
def add_captions(clip, segment):
    words = segment["text"].split()
    start_time = segment["start"]
    end_time = segment["end"]
    duration = end_time - start_time
    word_duration = duration / len(words)

    captions = []
    for i, word in enumerate(words):
        txt = TextClip(
            word,
            fontsize = 80.
            color = 'white',
            font = 'Arial-Bold',
            stroke_color = 'black',
            stroke_width = 2,
            method = 'caption'
        )
        txt = txt.set_position(('center', 1500)).set_start(start_time = i * word_duration).set_duration*(word_duration)
        captions.append(txt)
        
    return CompositeVideoClip([clip, *captions])

# Step 5 - Estabish Pipeline 
def process_video(video_path, output_path = "short_with_captions.mp4"):
    print("Transcribing video...")
    segments = transcribe_video(video_path) 
