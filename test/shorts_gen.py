import whisper
from moviepy.editor import *
from datetime import timedelta
from tqdm import tqdm
import os

VIDEO_PATH = "C:/Users/Arkou/Documents/AAL/RusselHeights/RussellHeightsHub_Teaser/RH_cine1.mp4"
CLIP_START = 0
CLIP_END = 10
SRT_PATH = "captions.srt"
OUTPUT_CLIP = "final_clip_moviepy.mp4"

# === STEP 1: Transcribe with Whisper ===
def transcribe_with_whisper(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path, word_timestamps=True)
    return result['segments']

# === STEP 2: Write SRT File ===
def write_srt(segments, srt_path):
    def format_time(seconds):
        td = timedelta(seconds=seconds)
        return str(td)[:12].replace('.', ',').zfill(12)

    with open(srt_path, "w", encoding="utf-8") as f:
        counter = 1
        for seg in segments:
            start = seg['start']
            end = seg['end']
            if start > CLIP_END:
                break
            if end < CLIP_START:
                continue
            # Clip trimming
            s = max(start, CLIP_START) - CLIP_START
            e = min(end, CLIP_END) - CLIP_START
            if s >= e:
                continue

            f.write(f"{counter}\n")
            f.write(f"{format_time(s)} --> {format_time(e)}\n")
            f.write(seg['text'].strip() + "\n\n")
            counter += 1

# === STEP 3: Parse SRT and Create Captions ===
def parse_srt_to_clips(srt_path, width=1080):
    from pysrt import open as open_srt
    subs = open_srt(srt_path)
    caption_clips = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end = sub.end.ordinal / 1000.0
        txt_clip = TextClip(sub.text, fontsize=60, color='white', font="Arial-Bold", stroke_color="black", stroke_width=2, size=(width * 0.9, None), method='caption')
        txt_clip = txt_clip.set_start(start).set_end(end).set_position(('center', 1500))
        caption_clips.append(txt_clip)
    return caption_clips

# === STEP 4: Create Vertical Blurred Video with Captions ===
def create_moviepy_video():
    clip = VideoFileClip(VIDEO_PATH).subclip(CLIP_START, CLIP_END)

    # Create background: resize + blur
    bg = clip.resize(height=1920).crop(width=1080, x_center=clip.w/2).fl_image(lambda f: blur_frame(f, 45))

    # Foreground: resize to fit width
    fg = clip.resize(width=1080)

    # Composite
    base = CompositeVideoClip([bg, fg.set_position("center")], size=(1080, 1920))

    # Add captions
    captions = parse_srt_to_clips(SRT_PATH)
    final = CompositeVideoClip([base, *captions])
    final.write_videofile(OUTPUT_CLIP, fps=30)

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Transcribing with Whisper...")
    segments = transcribe_with_whisper(VIDEO_PATH)

    print("📝 Generating SRT...")
    write_srt(segments, SRT_PATH)

    print("🎬 Rendering final video with MoviePy...")
    create_moviepy_video()

    print("✅ Done! Saved to:", OUTPUT_CLIP)
