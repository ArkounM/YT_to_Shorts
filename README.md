# YouTube to Shorts Converter

A Python script that automatically extracts short-form clips from long-form videos by detecting specific keywords and generating vertical (9:16) format videos suitable for platforms like YouTube Shorts, Instagram Reels, or TikTok.

## Features

- Automatically detects keywords in video audio
- Converts landscape videos to vertical 9:16 format
- Centers original video with blurred background
- Optional caption overlay with speech transcription
- Customizable clip duration
- Multiple output clips from a single video

## Prerequisites

```bash
pip install opencv-python numpy moviepy whisper textwrap3 Pillow
```

You'll also need FFmpeg installed on your system:
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

## Usage

Basic usage:
```bash
python instaClips.py "path/to/your/video.mp4"
```

Advanced options:
```bash
python instaClips.py "input_video.mp4" \
    --output-dir "output_folder" \
    --keyword "Day" \
    --min-duration 45 \
    --max-duration 60 \
    --captions
```

### Parameters

- `video_path`: Path to input video file
- `--output-dir`: Directory for output clips (default: "instagram_clips")
- `--keyword`: Word to detect for clip creation (default: "Day")
- `--min-duration`: Minimum clip duration in seconds (default: 60)
- `--max-duration`: Maximum clip duration in seconds (default: 60)
- `--captions`: Enable caption overlay (optional)

## Output

The script will:
1. Extract audio from the video
2. Transcribe the audio using Whisper
3. Find segments containing the keyword
4. Generate vertical format clips with:
   - Centered original video
   - Blurred background
   - Optional captions
5. Save clips as individual MP4 files

## Example

```bash
python instaClips.py "weekly_vlog.mp4" --keyword "today" --min-duration 45 --captions
```

This will create vertical clips whenever the word "today" is mentioned, with each clip being at least 45 seconds long and including captions.

## Notes

- Clip duration is centered around the detected keyword
- Captions are generated from speech transcription
- Output videos use H.264 codec for best compatibility
- Temporary files are automatically cleaned up

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests to improve the script.