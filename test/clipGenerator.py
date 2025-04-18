import whisper
import torch
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, ColorClip
from moviepy.video.fx import resize, crop
import numpy as np
from PIL import Image, ImageFilter
import os

def create_vertical_blur_background(frame, target_height, target_width):
    # Convert frame to PIL Image
    img = Image.fromarray(frame)
    
    # Create blurred background
    background = img.resize((target_width, target_height))
    background = background.filter(ImageFilter.GaussianBlur(radius=30))
    
    # Calculate scaling for main content
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = target_height
    new_width = int(new_height * aspect_ratio)
    
    if new_width > target_width:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    
    # Resize main content
    main_content = img.resize((new_width, new_height))
    
    # Center the main content
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    background.paste(main_content, (x_offset, y_offset))
    return np.array(background)

def process_video(video_path, output_dir="output_clips"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    video = VideoFileClip(video_path)
    
    # Load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device)
    
    # Transcribe audio
    result = model.transcribe(video_path)
    
    # Align words
    model_a, metadata = whisper.load_align_model(language_code=result["language"], device=device)
    result = whisper.align(result["segments"], model_a, metadata, video_path, device)
    
    # Find segments containing "Day"
    day_segments = []
    for segment in result["segments"]:
        for word in segment["words"]:
            if "day" in word["text"].lower():
                day_segments.append((segment, word))
    
    # Process each segment
    for idx, (segment, day_word) in enumerate(day_segments):
        # Extract clip with context (3 seconds before and after)
        start_time = max(0, day_word["start"] - 3)
        end_time = min(video.duration, day_word["end"] + 3)
        
        clip = video.subclip(start_time, end_time)
        
        # Set target dimensions (9:16 aspect ratio)
        target_width = 1080
        target_height = 1920
        
        # Create vertical version with blurred background
        vertical_clip = clip.fl_image(
            lambda frame: create_vertical_blur_background(frame, target_height, target_width)
        )
        
        # Create caption with highlighted keyword
        text_before = segment["text"][:day_word["text"].start()]
        text_after = segment["text"][day_word["text"].end():]
        
        # Create caption clips
        txt_clip = TextClip(
            segment["text"],
            fontsize=40,
            color='white',
            size=(target_width - 100, None),
            method='caption'
        ).set_position(('center', 'bottom')).set_duration(end_time - start_time)
        
        # Combine video and text
        final_clip = CompositeVideoClip([vertical_clip, txt_clip])
        
        # Write output
        output_path = os.path.join(output_dir, f"clip_{idx + 1}.mp4")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30
        )
        
        # Close clips
        final_clip.close()
        vertical_clip.close()
        
    # Clean up
    video.close()

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video path
    process_video(video_path)