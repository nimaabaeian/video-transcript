"""
Transcribe a video into a Markdown file, with speaker labels.

Usage:
    python transcribe.py my_video.mp4
    python transcribe.py my_video.mp4 --speakers 2 --model small
"""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm

SAMPLE_RATE = 16000

load_dotenv(Path(__file__).with_name(".env"))  # reads HF_TOKEN


def load_audio(path):
    """Decode any video/audio file to mono 16 kHz float32 using ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def transcribe(audio, model_name, language, device):
    """Return a list of (start, end, word) for every spoken word."""
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio, language=language, vad_filter=True, word_timestamps=True, condition_on_previous_text=False
    )
    words = []
    with tqdm(total=round(info.duration), unit="s", desc="Transcribing") as bar:
        for segment in segments:
            # Whisper sometimes invents text after the audio ends; skip it
            words += [(w.start, w.end, w.word) for w in segment.words if w.start < info.duration]
            bar.n = min(round(segment.end), bar.total)
            bar.refresh()
    return words


def diarize(audio, num_speakers, device):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN")
    )
    pipeline.to(torch.device(device))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    with ProgressHook() as hook:
        result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, num_speakers=num_speakers, hook=hook)
    return [(turn.start, turn.end, speaker) for turn, _, speaker in result.exclusive_speaker_diarization.itertracks(yield_label=True)]


def find_speaker(start, end, turns):
    """Return the speaker who talks the most during [start, end]."""
    overlap = {}
    for t_start, t_end, speaker in turns:
        overlap[speaker] = overlap.get(speaker, 0) + max(0, min(end, t_end) - max(start, t_start))
    return max(overlap, key=overlap.get) if overlap and max(overlap.values()) > 0 else None


def timestamp(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_markdown(title, words, turns):
    # Give speakers friendly names in order of appearance: Speaker 1, Speaker 2, ...
    names = {}
    blocks = []  # [start_time, speaker_name, [words]]
    for start, end, word in words:
        speaker = find_speaker(start, end, turns)
        if speaker is None:  # no speaker detected here: keep the previous speaker
            name = blocks[-1][1] if blocks else "Unknown"
        else:
            name = names.setdefault(speaker, f"Speaker {len(names) + 1}")
        if blocks and blocks[-1][1] == name:
            blocks[-1][2].append(word)
        else:
            blocks.append([start, name, [word]])

    lines = [f"# Transcript: {title}", ""]
    for start, name, block_words in blocks:
        lines += [f"**{name}** `[{timestamp(start)}]`", "", "".join(block_words).strip(), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Video -> Markdown transcript with speakers")
    parser.add_argument("video", type=Path, help="path to the video (or audio) file")
    parser.add_argument("-o", "--output", type=Path, help="output .md file (default: next to the video)")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model: tiny, base, small, medium, large-v3-turbo")
    parser.add_argument("--language", help="language code like 'en' (default: auto-detect)")
    parser.add_argument("--speakers", type=int, help="number of speakers, if you know it")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output or args.video.with_suffix(".md")

    print("Loading audio...")
    audio = load_audio(args.video)

    print(f"Transcribing on {device}...")
    words = transcribe(audio, args.model, args.language, device)

    print("Detecting speakers...")
    turns = diarize(audio, args.speakers, device)

    output.write_text(to_markdown(args.video.name, words, turns), encoding="utf-8")
    print(f"Done -> {output}")


if __name__ == "__main__":
    main()
