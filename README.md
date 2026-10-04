# video-transcribe

Transcribe a video or audio file into a Markdown file with speaker labels.

It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for word-level transcription and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/).
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with a Hugging Face token (needed for the pyannote models):

   ```
   HF_TOKEN=your_token_here
   ```

## Usage

```
python transcribe.py my_video.mp4
python transcribe.py my_video.mp4 --speakers 2 --model small
```

The transcript is written to a Markdown file next to the input.

## Models

`--model` accepts any faster-whisper model name:

| Size | Names |
|---|---|
| Tiny | `tiny`, `tiny.en` |
| Base | `base`, `base.en` |
| Small | `small`, `small.en` |
| Medium | `medium`, `medium.en` |
| Large | `large-v1`, `large-v2`, `large-v3`, `large` |
| Turbo | `large-v3-turbo`, `turbo` |
| Distilled (English only) | `distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3` |

Smaller models are faster but less accurate. `.en` models are English-only. You can also pass a Hugging Face repo ID or a local path to a converted model.
