import whisper
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, vfx
from datetime import timedelta
import os
import cv2
import numpy as np
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})  # Adjust path
def blur_frame(frame, ksize=45):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


# === CONFIG ===
VIDEO_PATH = "C:/Users/Arkou/Documents/AAL/RusselHeights/RussellHeightsHub_Teaser/RH_cine1.mp4"
CLIP_START = 0  # in seconds
CLIP_END = 10   # in seconds
OUTPUT_CLIP = "C:/Users/Arkou/Documents/AAL/RusselHeights/RussellHeightsHub_Teaser/final_clip.mp4"

# === STEP 1: Transcribe Video with Whisper ===
def transcribe_with_whisper(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path, word_timestamps=True)
    return result['segments']

# === STEP 2: Get Relevant Segment from Transcription ===
def extract_words_for_segment(segments, clip_start, clip_end):
    words = []
    for seg in segments:
        for word in seg.get('words', []):
            if clip_start <= word['start'] <= clip_end:
                words.append(word)
    return words

# === STEP 3: Create Blurred Vertical Video ===
def format_clip_vertical(video_path, start, end):
    clip = VideoFileClip(video_path).subclip(start, end)
    
    # Foreground: original resized to fit vertically
    fg = clip.resize(width=1080)

    # Background: blur stretched to 1920 height
    bg = clip.resize(height=1920).crop(width=1080, x_center=clip.w/2).fl_image(lambda f: blur_frame(f, 45))


    # Combine foreground and blurred background
    final = CompositeVideoClip([bg, fg.set_position("center")], size=(1080, 1920))
    return final

# === STEP 4: Add Captions Word-by-Word ===
def add_captions(clip, words, clip_start):
    caption_clips = []
    for word in words:
        start = word['start'] - clip_start
        end = word['end'] - clip_start
        txt = TextClip(
            word['word'],
            fontsize=80,
            color="white",
            font="Arial-Bold",
            stroke_color="black",
            stroke_width=2,
            method="label",
            size=(1000, None)
        ).set_position(("center", 1600)).set_start(start).set_duration(end - start)
        caption_clips.append(txt)

    return CompositeVideoClip([clip, *caption_clips], size=(1080, 1920))

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Transcribing...")
    segments = transcribe_with_whisper(VIDEO_PATH)

    print("💬 Extracting words for clip...")
    words = extract_words_for_segment(segments, CLIP_START, CLIP_END)

    print("🎞️ Formatting video...")
    clip = format_clip_vertical(VIDEO_PATH, CLIP_START, CLIP_END)

    print("💬 Adding captions...")
    final = add_captions(clip, words, CLIP_START)

    print("💾 Exporting final video...")
    final.write_videofile(OUTPUT_CLIP, fps=30, codec="libx264", audio_codec="aac")

    print("✅ Done!")
