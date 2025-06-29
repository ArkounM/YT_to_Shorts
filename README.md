# AI Video Processing Suite

This suite of tools leverages AI to process videos for various tasks, including transcription, captioning, chapter generation, and creating short clips for social media. It now features an optional Gradio-based graphical user interface (GUI) for ease of use.

## Core Features

-   **Advanced Transcription**: Utilizes Whisper for accurate speech-to-text.
    -   **GPU Acceleration**: Supports CUDA for faster transcription if a compatible NVIDIA GPU and PyTorch are installed.
    -   Device selection (`cpu`/`gpu`) via CLI or GUI.
-   **AI-Powered Content Generation**:
    -   **Chapter Generation**: Suggests YouTube chapter markers.
    -   **Clip Suggestions**: Identifies engaging moments for short clips.
-   **Flexible LLM Integration**: Supports multiple LLM providers:
    -   Google Gemini
    -   OpenAI (GPT models)
    -   OpenRouter (access to a wide variety of models)
    -   Deepseek
    -   Provider and model selection via CLI or GUI.
-   **Video Processing**:
    -   **Caption Rendering**: Burns captions onto videos with word-by-word highlighting and customizable styling.
    -   **Clip Creation**: Extracts selected segments and can format them (e.g., to vertical 9:16, though full video processing for clips in GUI is under development).
-   **User Interfaces**:
    -   **Command-Line Interface (CLI)**: Full functionality available through scripts like `generateCaptions.py`, `generateChapters.py`, `generateClips.py`.
    -   **Graphical User Interface (GUI)**: A Gradio app (`app.py`) provides access to most features, including file uploads, option selection, and API key management for the session.

## Prerequisites

### 1. Python Environment
Ensure you have Python 3.8+ installed.

### 2. FFmpeg
FFmpeg is crucial for audio extraction and video processing.
-   **Windows**: Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
-   **Linux**: `sudo apt update && sudo apt install ffmpeg`
-   **macOS**: `brew install ffmpeg`
    You can verify by typing `ffmpeg -version` in your terminal.

### 3. Python Packages
Install the required packages using pip:
```bash
pip install opencv-python numpy moviepy openai-whisper Pillow torch torchvision torchaudio requests google-generativeai openai gradio
```
-   `openai-whisper`: For transcription.
-   `torch`, `torchvision`, `torchaudio`: For PyTorch, used by Whisper. For GPU support, ensure your PyTorch installation is CUDA-enabled. You might need to install it separately from the [PyTorch website](https://pytorch.org/get-started/locally/) if you have specific CUDA version needs.
-   `requests`: For OpenRouter API calls.
-   `google-generativeai`: For Gemini API.
-   `openai`: For OpenAI and Deepseek APIs.
-   `gradio`: For the graphical user interface.
-   Other packages are for video/image manipulation and general utilities.

### 4. API Keys (Optional but Recommended for LLM features)
For chapter and clip generation features using Large Language Models (LLMs):
-   **Google Gemini**: Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey). Set `GEMINI_API_KEY` environment variable or provide via GUI/CLI.
-   **OpenAI**: Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys). Set `OPENAI_API_KEY` or provide.
-   **OpenRouter**: Get an API key from [OpenRouter.ai](https://openrouter.ai/keys). Set `OPENROUTER_API_KEY` or provide.
-   **Deepseek**: Get an API key from [Deepseek Platform](https://platform.deepseek.com/). Set `DEEPSEEK_API_KEY` or provide.

Setting API keys as environment variables is generally recommended for CLI use. The GUI also allows entering keys for the current session.

## Usage

### 1. Graphical User Interface (GUI)
The most user-friendly way to use the suite.
```bash
python app.py
```
Navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`). The GUI offers tabs for:
-   **Transcription & Captioning**: Upload video, choose Whisper model, device, styling, and get transcript and/or captioned video.
-   **Chapter Generation**: Upload video or transcript, select LLM, and generate YouTube chapters.
-   **Clip Generation**: Upload video or transcript, select LLM, and get suggestions for short clips. (Note: Full video processing for clips, like 9:16 formatting, is still primarily CLI or under active development in the GUI's backend).
-   **Settings**: Configure API keys for your current session.

### 2. Command-Line Interface (CLI)
Each script focuses on a specific task. Common arguments include:
-   `--device {cpu|gpu}`: For Whisper transcription.
-   `--llm-provider {gemini|openai|openrouter|deepseek}`: For LLM-based tasks.
-   `--llm-model <model_name>`: Specify a particular LLM model.
-   `--api-key <your_api_key>`: Pass API key directly.

**Example: Generating Captions**
```bash
python generateCaptions.py "path/to/video.mp4" --output-path "output/captioned_video.mp4" --whisper-model "base" --device "gpu" --font-path "/path/to/your/font.ttf" --bg-color "50,50,50,200"
```

**Example: Generating Chapters**
```bash
python generateChapters.py "path/to/video.mp4" --output-dir "output/chapters" --llm-provider "openai" --llm-model "gpt-3.5-turbo"
```

**Example: Generating Clips (Suggestions & Basic Extraction)**
```bash
python generateClips.py "path/to/video.mp4" --output-dir "output/clips" --llm-provider "openrouter" --llm-model "gryphe/mythomax-l2-13b" --captions
```
*(Refer to individual script `--help` for all parameters, e.g., `python generateClips.py --help`)*

### General CLI Parameters
Many scripts share parameters like:
-   `video_path`: Path to the input video file.
-   `--output-dir` or `--output-path`: Specifies where to save results.
-   `--whisper-model`: Transcription model size (e.g., "tiny", "base", "small", "medium", "large").
-   `--transcription-path`: Use an existing transcription JSON file.
-   `--skip-review` or `--no-review`: Bypass interactive review steps.
-   Caption styling arguments for `generateCaptions.py` and `generateClips.py` (e.g., `--font-path`, `--bg-color`, `--highlight-color`, `--text-color`).

## Output Structure
Output files are typically saved in the specified output directory or a subdirectory related to the input video name. This includes:
-   Transcribed text files or JSONs.
-   Captioned videos.
-   Chapter lists (text and JSON).
-   Generated video clips.
-   Metadata files (e.g., LLM suggestions).

## Modularity (For Developers)
The project has been refactored for better modularity. Core functionalities like video utilities, transcription, LLM communication, and caption rendering are now located in the `core/` directory. This makes it easier to maintain and extend the project.

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests to improve the script.
