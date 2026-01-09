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

  conda install -c conda-forge ffmpeg

  - Linux: `sudo apt-get install ffmpeg`
  - MacOS: `brew install ffmpeg`

### Linux-Specific Setup

If you're running on a **headless Linux server** (no GUI), you need to install system fonts for caption rendering:

```bash
# Install common fonts (choose one or more)
sudo apt-get install fonts-liberation      # Recommended
sudo apt-get install fonts-dejavu-core     # Alternative
sudo apt-get install fonts-ubuntu          # Alternative
```

The script will automatically detect and use available system fonts. It checks for fonts in common Linux locations:
- `/usr/share/fonts/truetype/liberation/`
- `/usr/share/fonts/truetype/dejavu/`
- `/usr/share/fonts/truetype/ubuntu/`

**Note:** The script now runs a dependency check on startup that will verify FFmpeg and fonts are properly installed.

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

## Troubleshooting

### Linux Issues

**Script runs but produces no output:**
1. Check that FFmpeg is installed: `ffmpeg -version`
2. Run the script - it will now perform a dependency check at startup
3. Install missing fonts if warned:
   ```bash
   sudo apt-get install fonts-liberation
   ```

**"No system font found" warning:**
- Captions will still work but may not look optimal
- Install fonts using the commands in the Linux-Specific Setup section above

**Virtual Environment:**
- You don't need to run from a virtual environment, but if you created one, make sure to activate it:
  ```bash
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows
  ```

**File paths:**
- Both Linux (`/home/user/video.mp4`) and Windows (`C:\Users\video.mp4`) paths are supported
- Use quotes around paths with spaces: `python generateClips.py "/path/with spaces/video.mp4"`

## Cross-Platform Support

This script now fully supports:
- ✅ Windows
- ✅ Linux (including headless servers)
- ✅ MacOS

The script automatically:
- Detects your operating system
- Finds appropriate system fonts
- Handles file paths correctly
- Checks dependencies on startup

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests to improve the script.
