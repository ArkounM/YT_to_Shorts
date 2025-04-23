# AI Video Clip Generator

An intelligent video processing tool that automatically identifies and extracts engaging moments from long-form videos, converting them into vertical format clips suitable for platforms like YouTube Shorts, Instagram Reels, or TikTok.

## Features

- 🎯 AI-powered clip detection using Google's Gemini API
- 🗣️ Automatic speech transcription using Whisper
- 📱 Converts to vertical 9:16 format
- 🎨 Professional styling:
  - Centered original video
  - Blurred background
  - Elegant caption design with rounded corners
  - Word-by-word highlighting
- ✂️ Interactive clip review and editing
- 🎬 Multiple output clips from a single video

## Prerequisites

### Required Python Packages
```bash
pip install opencv-python numpy moviepy openai-whisper Pillow google-generativeai textwrap3
```

### Required Software
- FFmpeg (for video processing)
  - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) or if you are using a conda env you can enter:

  conda install -c conda-forge ffmepg

  - Linux: `sudo apt-get install ffmpeg`
  - MacOS: `brew install ffmpeg`

### API Key
- Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Basic Usage
```bash
python generateClips.py "path/to/video.mp4"
```

### Advanced Options
```bash
python generateClips.py "input_video.mp4" \
    --output-dir "output_folder" \
    --min-clips 3 \
    --max-clips 8 \
    --whisper-model "base" \
    --api-key "your_gemini_api_key" \
    --captions \
    --no-review
```

### Parameters
- `video_path`: Path to input video file
- `--output-dir`: Output directory (default: "ai_clips")
- `--min-clips`: Minimum clips to generate (default: 3)
- `--max-clips`: Maximum clips to generate (default: 8)
- `--whisper-model`: Transcription model size ["tiny", "base", "small", "medium", "large"]
- `--api-key`: Google Gemini API key
- `--captions`: Enable captions (optional)
- `--no-review`: Skip clip review process (optional)
- `--bg-color`: Background color for captions in R,G,B,A format (default: 255,255,255,230)"
- `--highlight-color`: Highlight color for active words in R,G,B,A format (default: 255,226,165,220)
- `--text-color`: Text color in R,G,B format (default: 0,0,0)

## Output

The script generates:
1. Vertical format clips (9:16 aspect ratio)
2. JSON file with transcription
3. JSON file with clip suggestions
4. JSON file with final clip metadata
5. Clips with optional captions and word highlighting

## Interactive Review Mode

Unless `--no-review` is specified, you can:
- Review each suggested clip
- Edit transcriptions
- Adjust clip timings
- Approve/skip clips
- Preview clip content

## Caption Styling

- White rounded rectangle background
- Black text for readability
- Word-by-word highlighting
- Automatic positioning below the main video
- Smart line breaking for optimal display

## Files Generated

```
output_folder/
├── clip_1.mp4
├── clip_2.mp4
├── ...
├── transcription.json
├── clip_suggestions.json
└── clips_metadata.json
```

## Error Handling

- Graceful handling of missing FFmpeg
- Fallback for font loading
- API error management
- Temporary file cleanup

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests to improve the script.
